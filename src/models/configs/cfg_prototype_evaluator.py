# configs of prototype evaluator
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class PrototypeEvaluatorConfig:
    in_c: int  # input channels
    embed_dim: int  # embedding dimension
    patch_size : int # patch size
    roi_out_size: Tuple[int, int]  # output size for high feature maps
    spatial_scale : float  # spatial scale for high feature maps
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag


def build_prototype_evaluator_config(
    cfg: Dict      # global configuration
)->PrototypeEvaluatorConfig:

    return PrototypeEvaluatorConfig(
        in_c=cfg['MODEL']['IN_C'],
        embed_dim=cfg['MODEL']['EMBED_DIM'],
        patch_size=cfg['MODEL']['PATCH_SIZE'],
        roi_out_size=cfg['MODEL']['ROI_OUT_SIZE'],
        spatial_scale=cfg['MODEL']['SPATIAL_SCALE']
    )
