# configs of pseudo label generator
from dataclasses import dataclass
from typing import Tuple, List, Dict

from ...utils.vgg_layer_maps import vgg_layer_out_c_maps, vgg_layer_out_size_ratio_maps


@dataclass
class MPRCNNConfig:
    in_c: int # input channels
    num_classes: int # number of classes
    # -----for RPN-----
    rpn_anchor_sizes: Tuple[int]  # anchor sizes
    rpn_anchor_aspect_ratios: Tuple[float]  # anchor aspect ratios
    rpn_fg_iou_thresh: float  # foreground IoU threshold
    rpn_bg_iou_thresh: float  # background IoU threshold
    rpn_batch_size_per_image: int  # RPN batch size per image
    rpn_pre_nms_top_n: Dict[str, int]  # pre NMS top N, {"training": int, "testing": int}
    rpn_post_nms_top_n: Dict[str, int]  # post NMS top N, {"training": int, "testing": int}
    rpn_nms_thresh: float  # RPN NMS threshold
    # -----for PL generator-----
    # RoI Align mid to proposal
    roi_out_size_m2p: Tuple[int, int]  # output size
    spatial_scale_m2p: float  # spatial scale
    # -----for RoI Head-----
    hidden_dim: int     # MLP hidden dimension
    det_fg_iou_thresh: float  # foreground IoU threshold
    det_bg_iou_thresh: float  # background IoU threshold
    det_batch_size_per_image: int  # detection batch size per image
    det_positive_fraction: float  # detection positive fraction
    det_score_thresh: float  # detection score threshold
    det_nms_thresh: float   # detection NMS threshold
    detections_per_img: int # number of detections per image
    # RoI Align high to proposal
    roi_out_size_h2p: Tuple[int, int]  # output size for high feature maps to proposal box features
    spatial_scale_h2p: float  # spatial scale for high feature maps to proposal box features
    # RoI Align shared parameters
    sampling_ratio: int = 2  # sampling ratio
    aligned: bool = True  # aligned flag


def build_mp_rcnn_config(
    cfg: Dict      # global configuration
)->MPRCNNConfig:
    layer_indices = cfg['MODEL']['LAYER_INDICES']
    in_c = vgg_layer_out_c_maps[layer_indices[2]]
    spatial_scale_m2p = vgg_layer_out_size_ratio_maps[layer_indices[1]]
    spatial_scale_h2p = vgg_layer_out_size_ratio_maps[layer_indices[2]]
    rpn_anchor_sizes = tuple(cfg['MODEL']['RPN_ANCHOR_SIZES'])
    rpn_anchor_aspect_ratios = tuple(cfg['MODEL']['RPN_ANCHOR_ASPECT_RATIOS'])
    rpn_pre_nms_top_n=cfg['MODEL']['RPN_PRE_NMS_TOP_N']
    rpn_pre_nms_top_n = {
        "training" : rpn_pre_nms_top_n[0],
        "testing" : rpn_pre_nms_top_n[1]
    }
    rpn_post_nms_top_n=cfg['MODEL']['RPN_POST_NMS_TOP_N']
    rpn_post_nms_top_n = {
        "training" : rpn_post_nms_top_n[0],
        "testing" : rpn_post_nms_top_n[1]
    }


    return MPRCNNConfig(
        in_c=in_c,
        num_classes=cfg['DATA']['NUM_CLASSES'],
        rpn_anchor_sizes=rpn_anchor_sizes,
        rpn_anchor_aspect_ratios=rpn_anchor_aspect_ratios,
        rpn_fg_iou_thresh=cfg['MODEL']['RPN_FG_IOU_THRESH'],
        rpn_bg_iou_thresh=cfg['MODEL']['RPN_BG_IOU_THRESH'],
        rpn_batch_size_per_image=cfg['MODEL']['RPN_BATCH_SIZE_PER_IMAGE'],
        rpn_pre_nms_top_n=rpn_pre_nms_top_n,
        rpn_post_nms_top_n=rpn_post_nms_top_n,
        rpn_nms_thresh=cfg['MODEL']['RPN_NMS_THRESH'],
        roi_out_size_m2p=cfg['MODEL']['ROI_OUT_SIZE_MID'],
        spatial_scale_m2p=spatial_scale_m2p,
        hidden_dim=cfg['MODEL']['HIDDEN_DIM'],
        det_fg_iou_thresh=cfg['MODEL']['DET_FG_IOU_THRESH'],
        det_bg_iou_thresh=cfg['MODEL']['DET_BG_IOU_THRESH'],
        det_batch_size_per_image=cfg['MODEL']['DET_BATCH_SIZE_PER_IMAGE'],
        det_positive_fraction=cfg['MODEL']['DET_POSITIVE_FRACTION'],
        det_score_thresh=cfg['MODEL']['DET_SCORE_THRESH'],
        det_nms_thresh=cfg['MODEL']['DET_NMS_THRESH'],
        detections_per_img=cfg['MODEL']['DETECTIONS_PER_IMG'],
        roi_out_size_h2p=cfg['MODEL']['ROI_OUT_SIZE_H2P'],
        spatial_scale_h2p=spatial_scale_h2p
    )