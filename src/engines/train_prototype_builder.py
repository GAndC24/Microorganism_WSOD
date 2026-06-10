# train engine for Prototype Builder
import argparse
import torch
import yaml
from typing import Dict, Any
from pathlib import Path
from torchvision.transforms import v2 as T

from ..datasets.voc_dataset import build_voc_dataloader
from ..models.prototype_builder import build_prototype_builder_model
from ..utils.trainers.trainer_prototype_builder import build_prototype_builder_trainer


def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def train(
    config_path : str,
)-> None:

    cfg = _load_yaml(config_path)

    # set train seed
    torch.manual_seed(cfg['RUNTIME']["SEED"])
    torch.cuda.manual_seed(cfg['RUNTIME']["SEED"])

    # -----Init Dataloader-----
    # data preprocessing transforms
    img_size = cfg['DATA']["IMG_SIZE"]
    transform_aug = T.Compose([
        T.Resize((img_size, img_size)),
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
    train_loader = build_voc_dataloader(
        dataset_name=cfg['DATA']['DATASET_NAME'],
        split='train',
        target_mode='wb',
        batch_size=cfg['DATA']['BATCH_SIZE'],
        transforms=transform_aug,
    )

    # -----Init Model-----
    model = build_prototype_builder_model(cfg)

    # -----Init Trainer-----
    trainer = build_prototype_builder_trainer(model, train_loader, cfg)

    # -----Start Training-----
    trainer.train()