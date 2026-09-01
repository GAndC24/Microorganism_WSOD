'''
train engine for Prototype Builder
run:
    python -m src.engines.train_mp_rcnn --config "src/configs/cfg_mp_rcnn.yaml"
'''
import torch
import yaml
from typing import Dict, Any
from pathlib import Path
from torchvision.transforms import v2 as T
import argparse
import os


from ..datasets.voc_dataset import build_voc_dataloader
from ..models.mp_rcnn import build_mp_rcnn_model
from ..utils.trainers.trainer_mprcnn import build_mp_rcnn_trainer



def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training Config")

    p.add_argument('--config', type=str, required=True, help='config file path')
    

    return p.parse_args()


def train(
    config_path : str,
)-> None:

    cfg = _load_yaml(config_path)

    # set train seed
    torch.manual_seed(cfg['RUNTIME']["SEED"])
    torch.cuda.manual_seed(cfg['RUNTIME']["SEED"])

    # -----Init Dataloader-----
    # data preprocessing transforms
    img_w = cfg['DATA']["IMG_W"]
    img_h = cfg['DATA']["IMG_H"]
    img_size = (img_h, img_w)
    transform_aug = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_loader = build_voc_dataloader(
        dataset_name=cfg['DATA']['DATASET_NAME'],
        split='train',
        target_mode='wb',
        batch_size=cfg['DATA']['BATCH_SIZE'],
        transforms=transform_aug,
    )
    val_loader = build_voc_dataloader(
        dataset_name=cfg['DATA']['DATASET_NAME'],
        split='val',
        target_mode='gt',
        batch_size=cfg['DATA']['BATCH_SIZE'],
        transforms=transform_aug,
    )

    # -----Init Model-----
    model = build_mp_rcnn_model(cfg)

    # -----Load data-----
    dataset_MPs = torch.load(cfg['TRAINER']['DATASET_MPS_LOAD_PATH'])
    model.p_builder.load_state_dict(torch.load(cfg['TRAINER']['P_BUILDER_LOAD_PATH']))

    # -----Init Trainer-----
    trainer = build_mp_rcnn_trainer(
        model, 
        train_loader, 
        val_loader,
        dataset_MPs,
        cfg
    )

    # -----Start Training-----
    trainer.train()


def main():
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    args = _get_args()

    train(args.config)


if __name__ == '__main__':
    main()