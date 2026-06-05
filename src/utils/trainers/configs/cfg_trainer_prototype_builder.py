# configs of prototype builder trainer
from dataclasses import dataclass
from typing import Tuple, List, Dict
import torch

from ...logger import Logger


@dataclass
class PrototypeBuilderTrainerConfig:
    num_classes: int  # number of classes
    device: torch.device  # "cpu" or "cuda"
    epochs: int
    lr: float  # base lr
    warm_up_lr_factor: float  # min_lr = warm_up_lr_factor * lr
    warmup_epochs: int
    weight_decay: float
    model_save_path : str
    dataset_mps_save_path : str
    logger: Logger
    continue_train: bool = False
    checkpoint_path: str = None
    w_cam_loss : float = 0.5    # weight for cam loss
    w_patch_loss : float = 0.5  # weight for patch loss
    mp_ema_alpha: float = 0.99   # ema alpha for updating prototypes, [0.9, 0.99]


def build_prototype_builder_trainer_config(
    cfg: Dict      # global configuration
)->PrototypeBuilderTrainerConfig:
    if cfg['TRAINER']['CONTINUE_TRAIN']:
        checkpoint = torch.load(cfg['TRAINER']['CHECKPOINT_PATH'])
        logger = Logger(model_name=cfg['MODEL']['MODEL_NAME'], config=cfg, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name=cfg['MODEL']['MODEL_NAME'], config=cfg)


    return PrototypeBuilderTrainerConfig(
        num_classes=cfg['DATA']['NUM_CLASSES'],
        device=cfg['RUNTIME']['DEVICE'],
        epochs=cfg['TRAINER']['EPOCHS'],
        lr=cfg['TRAINER']['LR'],
        warm_up_lr_factor=cfg['TRAINER']['WARM_UP_LR_FACTOR'],
        warmup_epochs=cfg['TRAINER']['WARMUP_EPOCHS'],
        weight_decay=cfg['TRAINER']['WEIGHT_DECAY'],
        model_save_path=cfg['TRAINER']['MODEL_SAVE_PATH'],
        dataset_mps_save_path=cfg['TRAINER']['DATASET_MPS_SAVE_PATH'],
        logger=logger,
        continue_train=cfg['TRAINER']['CONTINUE_TRAIN'],
        checkpoint_path=cfg['TRAINER']['CHECKPOINT_PATH'],
        w_cam_loss=cfg['TRAINER']['W_CAM_LOSS'],
        w_patch_loss=cfg['TRAINER']['W_PATCH_LOSS'],
        mp_ema_alpha=cfg['TRAINER']['MP_EMA_ALPHA']
    )