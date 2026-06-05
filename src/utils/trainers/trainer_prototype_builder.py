# Prototype Builder Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm.auto import tqdm
from typing import Dict, List

from configs.cfg_trainer_prototype_builder import PrototypeBuilderTrainerConfig, build_prototype_builder_trainer_config
from ..losses.loss_funcs import get_patch_cls_loss
from ..losses.supcon_loss import SupConLossConfig, LossContrastMode, supervised_contrastive_loss


def _build_image_multi_hot(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
    """
    construct multi-hot labels for a batch of images
    :param targets: image annotations
    :param num_classes : number of classes
    :return: labels : multi-hot tensor, [B, num_classes]
    """
    labels = torch.zeros((len(targets), num_classes), dtype=torch.float32)
    for idx, target in enumerate(targets):
        if target["labels"].numel() == 0:
            continue
        labels[idx, target["labels"].unique()] = 1.0
    return labels


def _build_wb_one_hot(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
    """
    construct one-hot labels for a batch of weak boxes
    :param targets: image annotations
    :param num_classes : number of classes
    :return: labels : one-hot tensor, [R = num_wbb, num_classes]
    """
    all_labels = []
    for target in targets:
        all_labels.append(target["labels"])
    all_labels = torch.cat(all_labels, dim=0)  # [R]
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes), dtype=torch.float32).to(all_labels.device)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels


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

        self.dataset_MPs : Dict[int, torch.Tensor] = {}     # {class_id : prototype_embeddings_raw}

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
    def _update_dataset_mps_ema(self, batch_prototypes: Dict[int, torch.Tensor]) -> None:
        if (batch_prototypes is None) or (len(batch_prototypes) == 0):
            return
        alpha = float(self.cfg.mp_ema_alpha)
        for cls_id, p_new in batch_prototypes.items():
            if p_new is None:
                continue

            key = cls_id
            p_new = p_new.detach().to(self.cfg.device)
            p_new = torch.nn.functional.normalize(p_new, dim=-1, eps=1e-6)

            if key not in self.dataset_MPs:
                self.dataset_MPs[key] = p_new
            else:
                p_old = self.dataset_MPs[key].to(self.cfg.device)
                p_old = p_old.detach()

                p_updated = alpha * p_old + (1.0 - alpha) * p_new
                p_updated = torch.nn.functional.normalize(p_updated, dim=-1, eps=1e-6)

                self.dataset_MPs[key] = p_updated


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
            'cam' : 0.0,
            'patch' : 0.0,
            'patch_cls' : 0.0,
            'patch_supcon' : 0.0
        }
        for iter, (images, target) in pbar:
            images = [img.to(self.cfg.device) for img in images]
            targets = [{k: v.to(self.cfg.device) for k, v in t.items()} for t in target]

            wboxes = _build_wboxes(targets).to(self.cfg.device)
            wb_one_hot_labels = _build_wb_one_hot(targets, self.cfg.num_classes).to(self.cfg.device)
            X = torch.stack(images, dim=0)
            out = self.model(X, wboxes, wb_one_hot_labels)

            self._update_dataset_mps_ema(out['prototypes'])

            loss_patch_cls = get_patch_cls_loss(out['patch_logits'], wb_one_hot_labels)
            supcon_loss_config = SupConLossConfig()
            supcon_loss_config.contrast_mode = LossContrastMode.ONE_VIEW
            loss_patch_supcon = supervised_contrastive_loss(out['contrast_patch_features'], wb_one_hot_labels, supcon_loss_config)
            loss_patch = loss_patch_cls + loss_patch_supcon
            loss = self.cfg.w_cam_loss * out['loss_cam'] + self.cfg.w_patch_loss * loss_patch

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            epoch_losses['total'] += loss.item()
            epoch_losses['cam'] += out['loss_cam'].item()
            epoch_losses['patch'] += loss_patch.item()
            epoch_losses['patch_cls'] += loss_patch_cls.item()
            epoch_losses['patch_supcon'] += loss_patch_supcon.item()

            pbar.set_postfix({
                "Iter Loss: Total": f"{loss.item():.4f} ",
                "CAM": f"{out['loss_cam'].item():.4f} ",
                "Patch": f"{loss_patch.item():.4f} ",
                "lr": f"{self.optimizer.param_groups[0]['lr']}",
            })

        self.lr_scheduler.step()

        num_iters = len(self.train_loader)
        average_total_loss = epoch_losses['total'] / num_iters
        average_cam_loss = epoch_losses['cam'] / num_iters
        average_patch_loss = epoch_losses['patch'] / num_iters
        average_patch_cls_loss = epoch_losses['patch_cls'] / num_iters
        average_patch_supcon_loss = epoch_losses['patch_supcon'] / num_iters

        self.logger.add_info(
            f"Epoch [{epoch}/{self.cfg.epochs}]"
            f"Total Loss: {average_total_loss:.4f}, "
            f"CAM Loss: {average_cam_loss:.4f}\n"
            f"Patch Loss: {average_patch_loss:.4f} "
            f"Patch CLS Loss : {average_patch_cls_loss:.4f} "
            f"Patch SupCon Loss : {average_patch_supcon_loss:.4f} \n"
        )
        metrics = {
            'Epoch': epoch,
            'Total Loss': average_total_loss,
            'CAM Loss': average_cam_loss,
            'Patch Loss': average_patch_loss,
            'Patch CLS Loss': average_patch_cls_loss,
            'Patch SupCon Loss': average_patch_supcon_loss,
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


    def _save_model(self, model_save_path : str, model_name : str):
        model_state_dict = self.model.encoder.state_dict()
        model_file_path = f"{model_save_path}/{model_name}.pth"
        torch.save(model_state_dict, model_file_path)
        print(f"Model parameters saved to {model_file_path}")


    def _save_dataset_mps(self, dataset_mps_save_path : str):
        file_path = f"{dataset_mps_save_path}/dataset_MPs.pth"
        torch.save(self.dataset_MPs, file_path)
        print(f"Dataset morphological prototypes saved to {file_path}")


    def train(self)-> None:
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)
            checkpoints_save_path = self.logger.checkpoints_dir
            self._save_checkpoint(current_epoch=epoch, checkpoints_save_path=checkpoints_save_path)

        self.logger.end_train()
        model_save_path = self.cfg.model_save_path
        model_name = self.logger.model_name
        self._save_model(model_save_path, model_name)
        dataset_mps_save_path = self.cfg.dataset_mps_save_path
        self._save_dataset_mps(dataset_mps_save_path=dataset_mps_save_path)


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