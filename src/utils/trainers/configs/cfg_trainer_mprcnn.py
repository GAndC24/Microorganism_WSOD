# configs of MP RCNN trainer
from dataclasses import dataclass
from typing import Tuple, List, Dict
import torch

from ...logger import Logger


@dataclass
class MPRCNNTrainerConfig:
    num_classes: int  # number of classes
    device: torch.device  # "cpu" or "cuda"
    epochs: int
    lr: float  # base lr
    warm_up_lr_factor: float  # min_lr = warm_up_lr_factor * lr
    warmup_epochs: int
    weight_decay: float
    mp_ema_alpha: float
    logger: Logger
    continue_train: bool = False
    checkpoint_path: str = None
    


def build_mp_rcnn_trainer_config(
    cfg: Dict      # global configuration
)->MPRCNNTrainerConfig:
    if cfg['TRAINER']['CONTINUE_TRAIN']:
        checkpoint = torch.load(cfg['TRAINER']['CHECKPOINT_PATH'])
        logger = Logger(model_name=cfg['MODEL']['MODEL_NAME'], config=cfg, continue_existing=checkpoint['log_path'])
    else:       # new train
        logger = Logger(model_name=cfg['MODEL']['MODEL_NAME'], config=cfg)


    return MPRCNNTrainerConfig(
        num_classes=cfg['DATA']['NUM_CLASSES'],
        device=cfg['RUNTIME']['DEVICE'],
        epochs=cfg['TRAINER']['EPOCHS'],
        lr=cfg['TRAINER']['LR'],
        warm_up_lr_factor=cfg['TRAINER']['WARM_UP_LR_FACTOR'],
        warmup_epochs=cfg['TRAINER']['WARMUP_EPOCHS'],
        weight_decay=cfg['TRAINER']['WEIGHT_DECAY'],
        mp_ema_alpha=cfg['TRAINER']['MP_EMA_ALPHA'],
        logger=logger,
        continue_train=cfg['TRAINER']['CONTINUE_TRAIN'],
        checkpoint_path=cfg['TRAINER']['CHECKPOINT_PATH'],
    )