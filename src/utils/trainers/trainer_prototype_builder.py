# Prototype Builder Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.nn.functional as F

from .configs.cfg_trainer_prototype_builder import PrototypeBuilderTrainerConfig, build_prototype_builder_trainer_config
from ..losses.loss_funcs import get_constrain_loss, get_sep_loss
from ..vis_cams import visualize_cams


def _build_wb_one_hot(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    construct one-hot labels for a batch of weak boxes
    :param targets: image annotations
    :param num_classes : number of classes
    :return:
    - one_hot_labels: one-hot tensor, [R = num_wbb, num_classes]
    - all_labels: all labels, [R]
    """
    all_labels = []
    for target in targets:
        all_labels.append(target["labels"])
    all_labels = torch.cat(all_labels, dim=0)  # [R]
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes), dtype=torch.float32).to(all_labels.device)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels, all_labels


def _build_wboxes(targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Construct wboxes tensor from targets.
    :param targets: List of dictionaries containing bounding box information.
    :return: wboxes tensor of shape [R, 5], where each box is [batch_idx, x1, y1, x2, y2].
    """
    wboxes = []
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]  # Convert to [x1, y1, x2, y2]
        batch_indices = torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
        wboxes.append(torch.cat([batch_indices, boxes], dim=1))  # Combine batch_idx with boxes

    return torch.cat(wboxes, dim=0)  # Concatenate all boxes across the batch


class PrototypeBuilderTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        cfg: PrototypeBuilderTrainerConfig,
    )-> None:
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_loader = train_loader

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                # Linear warmup
                LinearLR(self.optimizer, start_factor=cfg.warm_up_lr_factor, end_factor=1.0),
                # cosine decay
                CosineAnnealingLR(self.optimizer, T_max=cfg.epochs - cfg.warmup_epochs,
                                  eta_min=cfg.lr * cfg.warm_up_lr_factor)
            ],
            milestones=[cfg.warmup_epochs]
        )

        self.start_epoch = 1

        self.logger = cfg.logger
        self.log_path = self.logger.log_dir

        self.dataset_MPs : Dict[int, torch.Tensor] = {}     # {class_id in [0, num_classes](0 for background) : prototype tensor}

        if cfg.continue_train:
            # load checkpoint
            checkpoint = torch.load(cfg.checkpoint_path)

            # load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully.")

            # load start epoch
            self.start_epoch = checkpoint['epoch'] + 1

            # load optimizer state
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            self.optimizer.load_state_dict(optimizer_state_dict)
            print("Optimizer state loaded successfully.")

            # load lr scheduler state
            lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
            print("Learning rate scheduler state loaded successfully.")

            # load dataset Morphological Prototypes
            self.dataset_MPs = checkpoint['dataset_MPs']
            print("Dataset MPs loaded successfully.")


    @torch.no_grad()
    def _initialize_missing_dataset_mps(
        self,
        batch_prototypes: Dict[int, torch.Tensor],
    ) -> None:
        """为尚未出现的类别建立零占位，避免写入无效 projector 输出。"""
        expected_keys = set(range(self.cfg.num_classes + 1))
        batch_prototypes_by_id = {
            int(class_id): prototype
            for class_id, prototype in batch_prototypes.items()
        }
        actual_keys = set(batch_prototypes_by_id)
        if actual_keys != expected_keys:
            raise ValueError(
                f"batch 原型 key 应为 {sorted(expected_keys)}，实际为 {sorted(actual_keys)}。"
            )

        for class_id in sorted(expected_keys):
            if class_id in self.dataset_MPs:
                continue

            prototype = batch_prototypes_by_id[class_id]
            if prototype is None:
                raise ValueError(f"类别 {class_id} 的 batch 原型不能为空。")

            self.dataset_MPs[class_id] = torch.zeros_like(
                prototype.detach(),
                device=self.cfg.device,
            )


    @torch.no_grad()
    def _update_dataset_mps_ema(
        self,
        batch_prototypes: Dict[int, torch.Tensor],
        valid_class_ids: torch.Tensor,
    ) -> None:
        if (batch_prototypes is None) or (len(batch_prototypes) == 0):
            return

        valid_foreground_ids = {
            int(class_id)
            for class_id in valid_class_ids.detach().flatten().cpu().tolist()
        }
        expected_foreground_ids = set(range(1, self.cfg.num_classes + 1))
        invalid_ids = valid_foreground_ids - expected_foreground_ids
        if invalid_ids:
            raise ValueError(f"有效前景类别超出范围：{sorted(invalid_ids)}。")

        self._initialize_missing_dataset_mps(batch_prototypes)

        # 背景始终更新；前景只更新当前 batch 实际出现的类别。
        update_class_ids = {0, *valid_foreground_ids}
        alpha = float(self.cfg.mp_ema_alpha)
        for cls_id, p_new in batch_prototypes.items():
            key = int(cls_id)
            if key not in update_class_ids:
                continue
            if p_new is None:
                raise ValueError(f"类别 {key} 的 batch 原型不能为空。")

            p_new = p_new.detach().to(self.cfg.device)
            p_new = F.normalize(p_new, dim=-1, eps=1e-6)
            p_old = self.dataset_MPs[key].detach().to(self.cfg.device)

            p_updated = alpha * p_old + (1.0 - alpha) * p_new
            self.dataset_MPs[key] = F.normalize(p_updated, dim=-1, eps=1e-6)


    def _train_one_epoch(self, epoch) -> None:
        num_iters = len(self.train_loader)
        pbar = tqdm(
            enumerate(self.train_loader, start=1),
            total=num_iters,
            desc=f"Epoch {epoch}/{self.cfg.epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        epoch_losses = {
            'total' : 0.0,
            'ccam' : 0.0,
            'constrain' : 0.0,
            'constrain_supcon' : 0.0,
            'constrain_proto' : 0.0,
            'sep' : 0.0,
            'sep_fg' : 0.0,
            'sep_bg' : 0.0,
        }
        for iter, (images, target) in pbar:
            images = [img.to(self.cfg.device) for img in images]
            targets = [
                {
                    k: v.to(self.cfg.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in target
            ]

            wboxes = _build_wboxes(targets).to(self.cfg.device)
            wb_one_hot_labels, all_labels = _build_wb_one_hot(targets, self.cfg.num_classes)
            wb_one_hot_labels, all_labels = wb_one_hot_labels.to(self.cfg.device), all_labels.to(self.cfg.device)
            X = torch.stack(images, dim=0)
            out = self.model(X, wboxes, wb_one_hot_labels)
            valid_class_ids = all_labels.unique() + 1

            if epoch == 1:
                self._update_dataset_mps_ema(
                    out['prototypes'],
                    valid_class_ids=valid_class_ids,
                )

            dataset_prototypes_dict = self.dataset_MPs
            proto_keys = sorted(dataset_prototypes_dict.keys())
            expected_proto_keys = list(range(self.cfg.num_classes + 1))
            if proto_keys != expected_proto_keys:
                raise ValueError(f"原型 key 应为 {expected_proto_keys}，实际为 {proto_keys}。")
            dataset_prototypes_dict = {
                key: F.normalize(
                    dataset_prototypes_dict[key].detach().to(self.cfg.device),
                    dim=-1,
                    eps=1e-6,
                )
                for key in proto_keys
            }
            contrast_patch_features_norm = F.normalize(out['contrast_patch_features'], dim=-1)
            loss_constrain_dict = get_constrain_loss(
                contrast_patch_features_norm,
                all_labels,
                dataset_prototypes_dict
            )

            batch_prototypes_dict = {}
            for idx, key in enumerate(proto_keys):
                dataset_prototypes_dict[key] = dataset_prototypes_dict[key].detach()
                batch_prototype = F.normalize(out['prototypes'][key], dim=-1)
                batch_prototypes_dict[key] = batch_prototype
            loss_sep_dict = get_sep_loss(
                dataset_prototypes=dataset_prototypes_dict,
                batch_prototypes=batch_prototypes_dict,
                valid_fg_class_ids=valid_class_ids,
            )

            loss = (self.cfg.w_ccam_loss * out['loss_ccam'] +
                    self.cfg.w_constrain_loss * loss_constrain_dict['loss_constrain'] +
                    self.cfg.w_sep_loss * loss_sep_dict['loss_sep'])

            if epoch > 1:
                self._update_dataset_mps_ema(
                    out['prototypes'],
                    valid_class_ids=valid_class_ids,
                )

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_losses['total'] += loss.item()
            epoch_losses['ccam'] += out['loss_ccam'].item()
            epoch_losses['constrain'] += loss_constrain_dict['loss_constrain'].item()
            epoch_losses['constrain_supcon'] += loss_constrain_dict['loss_constrain_supcon'].item()
            epoch_losses['constrain_proto'] += loss_constrain_dict['loss_constrain_proto'].item()
            epoch_losses['sep'] += loss_sep_dict['loss_sep'].item()
            epoch_losses['sep_fg'] += loss_sep_dict['loss_sep_fg'].item()
            epoch_losses['sep_bg'] += loss_sep_dict['loss_sep_bg'].item()

            pbar.set_postfix({
                "Iter Loss: Total": f"{loss.item():.4f} ",
                "CCAM": f"{out['loss_ccam'].item():.4f} ",
                "Constrain": f"{loss_constrain_dict['loss_constrain'].item():.4f} ",
                "Sep": f"{loss_sep_dict['loss_sep'].item():.4f} ",
                "lr": f"{self.optimizer.param_groups[0]['lr']}",
            })

            # # debug: visualize CCAMs
            # visualize_cams(
            #     ccams=out['ccams'],
            #     targets=targets,
            #     wb_one_hot_labels=wb_one_hot_labels,
            #     canvas_sizes=[
            #         tuple(int(value) for value in image.shape[-2:])
            #         for image in images
            #     ],
            #     epoch=epoch,
            #     iter=iter,
            # )


        self.lr_scheduler.step()

        num_iters = len(self.train_loader)
        average_total_loss = epoch_losses['total'] / num_iters
        average_ccam_loss = epoch_losses['ccam'] / num_iters
        average_constrain_loss = epoch_losses['constrain'] / num_iters
        average_constrain_supcon_loss = epoch_losses['constrain_supcon'] / num_iters
        average_constrain_proto_loss = epoch_losses['constrain_proto'] / num_iters
        average_sep_loss = epoch_losses['sep'] / num_iters
        average_sep_fg_loss = epoch_losses['sep_fg'] / num_iters
        average_sep_bg_loss = epoch_losses['sep_bg'] / num_iters

        self.logger.add_info(
            f"Epoch [{epoch}/{self.cfg.epochs}]\n"
            f"  Total Loss: {average_total_loss:.4f} | "
            f"CCAM Loss: {average_ccam_loss:.4f}\n"
            f"  Constrain Loss: {average_constrain_loss:.4f} | "
            f"SupCon Loss: {average_constrain_supcon_loss:.4f} | "
            f"Proto Loss: {average_constrain_proto_loss:.4f}\n"
            f"  Sep Loss: {average_sep_loss:.4f} | "
            f"FG Loss: {average_sep_fg_loss:.4f} | "
            f"BG Loss: {average_sep_bg_loss:.4f}\n\n"
        )
        metrics = {
            'Epoch': epoch,
            'Total Loss': average_total_loss,
            'CCAM Loss': average_ccam_loss,
            'Constrain Loss': average_constrain_loss,
            'Constrain SupCon Loss': average_constrain_supcon_loss,
            'Constrain Proto Loss': average_constrain_proto_loss,
            'Sep Loss': average_sep_loss,
            'Sep FG Loss': average_sep_fg_loss,
            'Sep BG Loss': average_sep_bg_loss,
        }
        self.logger.add_metrics(metrics)


    def _save_checkpoint(self, current_epoch : int, checkpoints_save_path : str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "epoch": current_epoch,
            "log_path": self.log_path,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "dataset_MPs": self.dataset_MPs,
        }

        file_path = f"{checkpoints_save_path}/checkpoint_{current_epoch}.pth"
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")


    def train(self)-> None:
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)
            checkpoints_save_path = self.logger.checkpoints_dir
            self._save_checkpoint(current_epoch=epoch, checkpoints_save_path=checkpoints_save_path)

        self.logger.end_train()


def build_prototype_builder_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: Dict   # global configuration
)-> PrototypeBuilderTrainer:
    trainer_cfg = build_prototype_builder_trainer_config(cfg)

    return PrototypeBuilderTrainer(
        model=model,
        train_loader=train_loader,
        cfg=trainer_cfg
    )
