# Prototype Evaluator
import torch
import torch.nn as nn
from typing import Dict, Any
from torchvision.ops import RoIAlign
from timm.models.vision_transformer import PatchEmbed
import torch.nn.functional as F
from torchvision.models import vgg16

from .configs.cfg_prototype_evaluator import PrototypeEvaluatorConfig, build_prototype_evaluator_config


class PrototypeEvaluator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # Default VGG-16 with aligned weights
        patch_embed : PatchEmbed,       # Patch Embed layer from Stage1
        config : PrototypeEvaluatorConfig       # PrototypeChecker configuration
    )-> None:
        super(PrototypeEvaluator, self).__init__()

        self.encoder = backbone
        self.config = config

        self.roi_align = RoIAlign(
            output_size=config.roi_out_size,
            spatial_scale=config.spatial_scale,
            sampling_ratio=config.sampling_ratio,
            aligned=config.aligned,
        )

        self.patch_embed = patch_embed


    def forward(
        self,
        x: torch.Tensor,        # input images, [B, C, H, W]
        boxes: torch.Tensor,        # GT boxes for RoI Align, [R, 5], for each box, [batch_idx, x1, y1, x2, y2]
        boxes_labels : torch.Tensor,        # class labels for GT boxes, [R, num_classes]
        prototypes : Dict[int, torch.Tensor],        # raw, {class_id, prototype tensor}
        return_details: bool = True,
        lse_alpha : float = 10.0,
        lse_eps : float = 1e-6
    ) -> Dict[str, Any]:
        '''
        :return: Similarity of GT and prototypes, {class_id : average_similarity}
        '''
        # get GT patch features
        self.encoder[-1] = nn.Identity()
        feature_maps = self.encoder(x)
        roi_features = self.roi_align(feature_maps, boxes)  # [R, C, H, W]
        patch_features = self.patch_embed(roi_features)  # [R, num_patches, D]

        # sort by class_id
        class_ids = sorted(list(prototypes.keys()))
        # norm
        proto_list = [prototypes[k] for k in class_ids]
        proto_norm_list = []
        for proto in proto_list:
            proto_norm = F.normalize(proto, dim=-1)
            proto_norm_list.append(proto_norm)
        proto_mat = torch.stack(proto_norm_list, dim=0)  # [num_classes, D]


        R, num_classes = boxes_labels.shape
        sims_sum = {k: 0.0 for k in range(num_classes)}
        sims_cnt = {k: 0 for k in range(num_classes)}

        # statistics: results details
        details = {
            "pos": {k: [] for k in range(num_classes)},  # sim_pos of each GT
            "neg_max": {k: [] for k in range(num_classes)},  # max sim_neg of each GT
            "margin": {k: [] for k in range(num_classes)},  # margin = sim_pos - max sim_neg
        }

        for i in range(R):
            gt_label = torch.argmax(boxes_labels[i]).item()

            gt_patch = patch_features[i]  # [num_patches, D]
            gt_patch = F.normalize(gt_patch, dim=1)

            # # mean pooling
            # gt_vec = gt_patch.mean(dim=0, keepdim=True)  # [1, D]
            # gt_vec = F.normalize(gt_vec, dim=1)  # [1, D]

            # LogSumExp pooling
            m = gt_patch.max(dim=0, keepdim=True).values  # [1, D]
            lse = m + torch.log(torch.exp(lse_alpha * (gt_patch - m)).mean(dim=0, keepdim=True) + lse_eps) / lse_alpha
            gt_vec = F.normalize(lse, dim=1, eps=lse_eps)  # [1, D]

            # similarity of GT and prototypes(all classes)
            sims_all = torch.matmul(gt_vec, proto_mat.t()).squeeze(0)  # [num_classes]

            sim_pos = sims_all[gt_label].item()     # positive class similarity

            # max negative class similarity
            if num_classes > 1:
                mask = torch.ones(num_classes, dtype=torch.bool, device=sims_all.device)
                mask[gt_label] = False
                sim_neg_max = sims_all[mask].max().item()
            else:
                sim_neg_max = float("-inf")

            margin = sim_pos - sim_neg_max if sim_neg_max != float("-inf") else float("inf")

            sims_sum[gt_label] += sim_pos
            sims_cnt[gt_label] += 1

            if return_details:
                details["pos"][gt_label].append(sim_pos)
                details["neg_max"][gt_label].append(sim_neg_max)
                details["margin"][gt_label].append(margin)

        out = {
            "sum": sims_sum,
            "cnt": sims_cnt,
        }
        if return_details:
            out["details"] = details

        return out


def build_prototype_evaluator(
    cfg: Dict # global configuration
)->PrototypeEvaluator:
    prototype_evaluator_config = build_prototype_evaluator_config(cfg)

    backbone = vgg16(pretrained=False).features
    backbone.load_state_dict(torch.load(cfg['MODEL']['BACKBONE_WEIGHTS_PATH']))

    patch_embed = PatchEmbed(
        img_size=prototype_evaluator_config.roi_out_size,
        patch_size=prototype_evaluator_config.patch_size,
        in_chans=prototype_evaluator_config.in_c,
        embed_dim=prototype_evaluator_config.embed_dim,
    )
    patch_embed.load_state_dict(torch.load(cfg['MODEL']['PATCH_EMBED_WEIGHTS_PATH']))

    return PrototypeEvaluator(
        backbone=backbone,
        patch_embed=patch_embed,
        config=prototype_evaluator_config
    )