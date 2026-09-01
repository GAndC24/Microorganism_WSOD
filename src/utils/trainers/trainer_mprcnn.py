# MP RCNN Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.nn.functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .configs.cfg_trainer_mprcnn import MPRCNNTrainerConfig, build_mp_rcnn_trainer_config
from ..vis_boxes import select_batch_highest_score_indices, visualize


# def _build_image_multi_hot(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
#     """
#     construct multi-hot labels for a batch of images
#     :param targets: image annotations
#     :param num_classes : number of classes
#     :return: labels : multi-hot tensor, [B, num_classes]
#     """
#     labels = torch.zeros((len(targets), num_classes), dtype=torch.float32)
#     for idx, target in enumerate(targets):
#         if target["labels"].numel() == 0:
#             continue
#         labels[idx, target["labels"].unique()] = 1.0
#     return labels


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



class MPRCNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_MPs: Dict[int, torch.Tensor],
        cfg: MPRCNNTrainerConfig,
    )-> None:
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_MPs = dataset_MPs  # {class_id in [0, num_classes](0 for background) : prototype tensor}

        self.amp_enabled = cfg.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

        trainable_params = (
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
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

            # load AMP scaler
            scaler_state = checkpoint["scaler_state_dict"]
            if scaler_state is not None:
                self.scaler.load_state_dict(scaler_state)
                print("AMP scaler loaded successfully.")


    # @torch.no_grad()
    # def _update_dataset_mps_ema(self, batch_prototypes: Dict[int, torch.Tensor]) -> None:
    #     if (batch_prototypes is None) or (len(batch_prototypes) == 0):
    #         return
    #     alpha = float(self.cfg.mp_ema_alpha)
    #     for cls_id, p_new in batch_prototypes.items():
    #         if p_new is None:
    #             continue
        
    #         key = cls_id
    #         p_new = p_new.detach().to(self.cfg.device)
    #         p_new = torch.nn.functional.normalize(p_new, dim=-1, eps=1e-6)
        
    #         if key not in self.dataset_MPs:
    #             self.dataset_MPs[key] = p_new
    #         else:
    #             p_old = self.dataset_MPs[key].to(self.cfg.device)
    #             p_old = p_old.detach()
        
    #             p_updated = alpha * p_old + (1.0 - alpha) * p_new
    #             p_updated = torch.nn.functional.normalize(p_updated, dim=-1, eps=1e-6)
        
    #             self.dataset_MPs[key] = p_updated


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
            'rpn' : 0.0,
            'det' : 0.0,
            'rpn_objectness' : 0.0,
            'rpn_box_reg' : 0.0,
            'det_classifier' : 0.0,
            'det_box_reg' : 0.0
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

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                out = self.model(
                    'train',
                    X, 
                    self.dataset_MPs,
                    wboxes, 
                    wb_one_hot_labels
                )   

                rpn_losses_dict = out['rpn_losses_dict']
                det_losses_dict = out['det_losses_dict']
                loss_rpn = rpn_losses_dict['loss_objectness'] + rpn_losses_dict['loss_rpn_box_reg']
                loss_det = det_losses_dict['loss_classifier'] + det_losses_dict['loss_box_reg']
                loss = loss_rpn + loss_det

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


            # debug
            if iter % 40 == 0:
                batch_pseudo_labels = self.model.batch_pseudo_labels
                B = len(batch_pseudo_labels)
                batch_highlighted_indices = select_batch_highest_score_indices(
                    [pseudo_labels["boxes"] for pseudo_labels in batch_pseudo_labels],
                    [pseudo_labels["scores"] for pseudo_labels in batch_pseudo_labels],
                    [pseudo_labels["labels"] for pseudo_labels in batch_pseudo_labels],
                )
                self.logger.add_info(
                    "\n"
                    "-----------------------------------------------------------------------------\n"
                    f"Epoch [{epoch}/{self.cfg.epochs}], Iter [{iter}/{num_iters}], Pseudo Labels:\n"
                    "-----------------------------------------------------------------------------\n"
                    "\n"
                )
                for i in range(B):
                    pseudo_labels = batch_pseudo_labels[i]
                    img_id = targets[i]["image_id"]
                    boxes = pseudo_labels['boxes']
                    scores = pseudo_labels['scores']
                    labels = pseudo_labels['labels']
                    self.logger.add_info(
                        f"\nImage ID: {img_id}\n"
                    )
                    n = boxes.shape[0]
                    for j in range(n):
                        self.logger.add_info(
                            f"\nbox: {boxes[j]}, score: {scores[j]}, label: {labels[j]}\n"
                        )
                    visualize(
                        img_id,
                        boxes,
                        scores,
                        labels,
                        self.logger.log_dir,
                        epoch,
                        iter,
                        highlighted_box_indices=batch_highlighted_indices[i],
                        box_canvas_size=tuple(images[i].shape[-2:]),
                    )


            epoch_losses['total'] += loss.item()
            epoch_losses['rpn'] += loss_rpn.item()
            epoch_losses['det'] += loss_det.item()
            epoch_losses['rpn_objectness'] += rpn_losses_dict['loss_objectness'].item()
            epoch_losses['rpn_box_reg'] += rpn_losses_dict['loss_rpn_box_reg'].item()
            epoch_losses['det_classifier'] += det_losses_dict['loss_classifier'].item()
            epoch_losses['det_box_reg'] += det_losses_dict['loss_box_reg'].item()

            pbar.set_postfix({
                "Iter Loss: Total": f"{loss.item():.4f} ",
                "RPN": f"{loss_rpn.item():.4f} ",
                "DET": f"{loss_det.item():.4f} ",
                "lr": f"{self.optimizer.param_groups[0]['lr']}",
            })

        self.lr_scheduler.step()

        num_iters = len(self.train_loader)
        average_total_loss = epoch_losses['total'] / num_iters
        average_rpn_loss = epoch_losses['rpn'] / num_iters
        average_det_loss = epoch_losses['det'] / num_iters
        average_rpn_objectness_loss = epoch_losses['rpn_objectness'] / num_iters
        average_rpn_box_reg_loss = epoch_losses['rpn_box_reg'] / num_iters
        average_det_classifier_loss = epoch_losses['det_classifier'] / num_iters
        average_det_box_reg_loss = epoch_losses['det_box_reg'] / num_iters

        self.logger.add_info(
            f"\nEpoch [{epoch}/{self.cfg.epochs}]"
            f"Total Loss: {average_total_loss:.4f}, "
            f"RPN Loss: {average_rpn_loss:.4f}, "
            f"DET Loss: {average_det_loss:.4f}\n"
            f"RPN Objectness Loss: {average_rpn_objectness_loss:.4f}, "
            f"RPN Box Reg Loss: {average_rpn_box_reg_loss:.4f}, "
            f"DET Classifier Loss: {average_det_classifier_loss:.4f}, "
            f"DET Box Reg Loss: {average_det_box_reg_loss:.4f}\n"
        )
        metrics = {
            'Epoch': epoch,
            'Total Loss': average_total_loss,
            'RPN Loss': average_rpn_loss,
            'DET Loss': average_det_loss,
            'RPN Objectness Loss': average_rpn_objectness_loss,
            'RPN Box Reg Loss': average_rpn_box_reg_loss,
            'DET Classifier Loss': average_det_classifier_loss,
            'DET Box Reg Loss': average_det_box_reg_loss,
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
            "scaler_state_dict": self.scaler.state_dict(),
        }

        file_path = f"{checkpoints_save_path}/checkpoint_{current_epoch}.pth"
        torch.save(checkpoint, file_path)
        print(f"Checkpoint saved to {file_path}")


    def _eval(self, epoch: int)-> None:
        loader = self.val_loader
        self.model.eval()
        map_metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=[0.5],   # mAP[@0.5]
            max_detection_thresholds=[1, 10, 100],
        ).to(self.cfg.device)

        num_iters = len(loader)
        pbar = tqdm(
            enumerate(loader, start=1),
            total=num_iters,
            desc=f"Computing Val mAP@[0.5]",
            leave=False,
            dynamic_ncols=True,
        )

        with torch.no_grad():
            for _, (X, targets) in pbar:
                X = [x.to(self.cfg.device) for x in X]
                X = torch.stack(X, dim=0)

                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.amp_enabled,
                ):
                    detections = self.model("inference", X)

                shifted_targets = []
                for t in targets:
                    t['boxes'] = t['boxes'].to(self.cfg.device)
                    t['labels'] = t['labels'].to(self.cfg.device)
                    shifted_targets.append({
                        "boxes": t["boxes"],
                        "labels": t["labels"] + 1   # shift to [1, num_classes], 0 for bg
                    })
                
                map_metric.update(detections, shifted_targets)
        
        map_value = float(map_metric.compute()['map'])

        self.logger.add_info(f"\nEpoch [{epoch}/{self.cfg.epochs}] Val mAP@[0.5]: {map_value:.4f}\n")
        self.logger.add_metrics({
            'Epoch': epoch,
            'Val mAP@[0.5]': map_value
        })


    def train(self)-> None:
        for epoch in range(self.start_epoch, self.cfg.epochs + 1):
            self.model.train()
            self._train_one_epoch(epoch)
            self._eval(epoch)
            checkpoints_save_path = self.logger.checkpoints_dir
            self._save_checkpoint(current_epoch=epoch, checkpoints_save_path=checkpoints_save_path)

        self.logger.end_train()


def build_mp_rcnn_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_MPs: Dict, 
    cfg: Dict   # global configuration
)-> MPRCNNTrainer:
    trainer_cfg = build_mp_rcnn_trainer_config(cfg)

    return MPRCNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_MPs=dataset_MPs,
        cfg=trainer_cfg
    )
