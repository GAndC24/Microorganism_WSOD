# configs of prototype builder
from dataclasses import dataclass
from typing import Tuple, List, Dict

from ...utils.vgg_layer_maps import vgg_layer_out_c_maps, vgg_layer_out_size_ratio_maps


@dataclass
class PrototypeBuilderConfig:
    embed_dim: int  # embedding dimension
    hidden_dim: int  # MLP hidden dimension
    layer_indices: List[int]  # feature layer indices, [low, mid, high]
    # for Morphological Prototype Generator
    in_c: int  # input channels
    patch_size: int  # patch size
    # for RoI Align
    roi_out_size_mid: Tuple[int, int]  # output size for middle feature maps
    spatial_scale_mid: float  # spatial scale for middle feature maps
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag


def build_prototype_builder_config(
    cfg: Dict      # global configuration
)->PrototypeBuilderConfig:
    layer_indices = cfg['MODEL']['LAYER_INDICES']
    in_c = vgg_layer_out_c_maps[layer_indices[1]]
    spatial_scale_mid = vgg_layer_out_size_ratio_maps[layer_indices[1]]

    return PrototypeBuilderConfig(
        embed_dim=cfg['MODEL']['EMBED_DIM'],
        hidden_dim=cfg['MODEL']['HIDDEN_DIM'],
        layer_indices=layer_indices,
        in_c=in_c,
        patch_size=cfg['MODEL']['PATCH_SIZE'],
        roi_out_size_mid=cfg['MODEL']['ROI_OUT_SIZE_MID'],
        spatial_scale_mid=spatial_scale_mid,
    )