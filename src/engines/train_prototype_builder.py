# train engine for Prototype Builder
import argparse
import os
import torch
import yaml
from typing import Dict, Any
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T



def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def _get_args()-> argparse:
    p = argparse.ArgumentParser(description="Training Config")

    # -----config file path-----
    p.add_argument('--config', type=str, help='Path to the config file.')


    return p.parse_args()


def main()-> None:
    os.environ["OMP_NUM_THREADS"] = "1"

    args = _get_args()

    cfg = _load_yaml(args.config)

    # set train seed
    torch.manual_seed(cfg['RUNTIME']["SEED"])
    torch.cuda.manual_seed(cfg['RUNTIME']["SEED"])

    # data configurations

    data_root = cfg["DATA"]['DATA_ROOT']


    # Init Dataset
    # data preprocessing transforms
    transform_aug = T.Compose([
        T.Resize(cfg["DATA"]['IMAGE_SIZE']),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # # 轻度几何扰动
        # T.RandomHorizontalFlip(p=0.5),
        # T.RandomVerticalFlip(p=0.5),
        # T.RandomRotation(degrees=15),
        # # 亮度/对比度扰动
        # T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25)], p=0.8),
        # T.RandomAutocontrast(p=0.2),
        # T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.2),
    ])

    train_dataset = UrinarySedimentDataset(root=data_root, split="train", transforms=transform_aug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=detection_collate_fn)

    # Initialize Model
    in_c = vgg_layer_out_c_maps[layer_indices[1]]
    spatial_scale_mid = vgg_layer_out_size_ratio_maps[layer_indices[1]]
    spatial_scale_high = vgg_layer_out_size_ratio_maps[layer_indices[2]]
    stage1_config = Stage1Config(
        num_classes=num_classes,
        embed_dim=embed_dim,
        img_size=img_size,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        layer_indices=layer_indices,
        mask_threshold=mask_threshold,
        gaussian_sigma=gaussian_sigma,
        in_c=in_c,
        patch_size=patch_size,
        components_range=components_range,
        random_state=random_state,
        max_iter=max_iter,
        roi_out_size_mid=roi_out_size_mid,
        roi_out_size_high=roi_out_size_high,
        spatial_scale_mid=spatial_scale_mid,
        spatial_scale_high=spatial_scale_high,
        sampling_ratio=sampling_ratio,
    )
    model = build_Stage1_model(stage1_config)

    # Initialize Trainer
    supcon_loss_config = SupConLossConfig(
        temperature=temperature,
        positives_cap=positives_cap
    )

    if continue_train:
        checkpoint = torch.load(checkpoint_path)
        logger = Logger(model_name="Stage1", config=config, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name="Stage1", config=config)

    stage1_trainer_config = Stage1TrainerConfig(
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        lr=lr,
        warm_up_lr_factor=warm_up_lr_factor,
        warmup_epochs=warmup_epochs,
        weight_decay=weight_decay,
        checkpoints_save_path=checkpoints_save_path,
        model_save_path=model_save_path,
        dataset_mps_save_path=dataset_mps_save_path,
        logger=logger,
        continue_train=continue_train,
        checkpoint_path=checkpoint_path,
        w_img_loss=w_img_loss,
        w_wbb_loss=w_wbb_loss,
        w_cam_loss=w_cam_loss,
        w_patch_loss=w_patch_loss,
        mp_ema_alpha=mp_ema_alpha
    )
    trainer = build_stage1_trainer(model, train_loader, stage1_trainer_config, supcon_loss_config)

    # Start Training
    trainer.train()


if __name__ == "__main__":
    main()