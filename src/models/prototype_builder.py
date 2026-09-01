# Prototype Builder, only construct morphological prototypes
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from torchvision.ops import RoIAlign
from timm.models.vision_transformer import PatchEmbed
import fvcore.nn.weight_init as weight_init
from torchvision.models import vgg16
from torchvision.transforms import v2 as T

from .configs.cfg_prototype_builder import PrototypeBuilderConfig, build_prototype_builder_config
from ..utils.losses.ccam_loss import SimMaxLoss, SimMinLoss


# -----Feature Hook-----
# Extract feature maps from given layer of backbone
class FeatureHook:
    def __init__(self)-> None:
        self.outputs = OrderedDict()
        self.handles = []

    def _hook(self, name : str)-> callable:
        """
        :param name: feature name, such as 'low, mid, high'
        :return: function fn
        """
        def fn(module : nn.Module, inp : Tuple, out : torch.Tensor)-> None:
            self.outputs[name] = out
        return fn

    def register(self, module: nn.Module, name: str)-> None:
        handle = module.register_forward_hook(self._hook(name))
        self.handles.append(handle)

    def clear(self):
        """
        clear hook
        """
        self.outputs.clear()

    def remove(self):
        """
        remove hook
        """
        for h in self.handles:
            h.remove()
        self.handles = []


# Build backbone hook
def build_backbone_hook(backbone : nn.Module, indices : List[int]) -> FeatureHook:
    """
    :param backbone: Default VGG-16 backbone
    :param indices: feature layer indices
    :return: FeatureHook object
    """
    hook = FeatureHook()
    for idx, tag in zip(indices, ['low', 'mid', 'high']):
        module = backbone[idx]
        hook.register(module, tag)
    return hook


# Build VGG-16 backbone with hook
def build_vgg16_backbone_with_hook(indices : List[int]) -> Tuple[nn.Module, FeatureHook]:
    """
    :return: VGG-16 backbone and FeatureHook object
    """
    # Load default VGG-16 backbone
    backbone = vgg16(pretrained=True).features
    # Build backbone hooker
    hook = build_backbone_hook(backbone, indices)
    return backbone, hook


# -----Feature Augmentation Transform-----
class FeatureMapTransform(nn.Module):
    def __init__(
        self,
        num_views: int = 2,
        mask_prob: float = 0.5,
        mask_scale: Tuple[float, float] = (0.02, 0.20),
        noise_prob: float = 0.5,
        noise_sigma: float = 0.05,
        keep_original: bool = False,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.keep_original = keep_original

        self.transform = T.Compose([
            T.RandomErasing(
                p=mask_prob,
                scale=mask_scale,
                ratio=(0.3, 3.3),
                value=0.0,
                inplace=False,
            ),
            T.RandomApply([
                T.GaussianNoise(
                    mean=0.0,
                    sigma=noise_sigma,
                    clip=False,
                )
            ], p=noise_prob),
        ])

    def forward(
        self,
        x: torch.Tensor     # shape [R=num_wbs, C, H, W]
    )->torch.Tensor:
        '''
        :return: shape [R, V=num_views, C, H, W]
        '''
        views = []

        if self.keep_original:
            views.append(x)

        while len(views) < self.num_views:
            views.append(self.transform(x))

        return torch.stack(views, dim=1)


# -----CCAM Generator-----
class CCAMGenerator(nn.Module):
    def __init__(
        self,
        in_c : int,     # input channels
        alpha : float = 0.05
    )-> None:
        super(CCAMGenerator, self).__init__()

        self.activation_head = nn.Conv2d(in_c, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.criterion = [
            SimMaxLoss(metric='cos', alpha=alpha), # BG-BG positive contrast
            SimMinLoss(metric='cos'),   # BG-FG negative contrast
            SimMaxLoss(metric='cos', alpha=alpha)   # FG-FG positive contrast
        ]


    def forward(
        self,
        x : torch.Tensor    # input feature maps, [N, C, H, W]
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return:
            ccam(torch.Tensor): class activation map, [N, 1, H, W]
            loss_ccam(torch.Tensor): CCAM loss
        """
        N, C, H, W = x.size()

        ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]

        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        fg_feats = fg_feats.reshape(x.size(0), -1)      # [N, C]
        bg_feats = bg_feats.reshape(x.size(0), -1)      # [N, C]
        for loss in self.criterion:
            loss.to(x.device)

        loss_bg_bg = self.criterion[0](bg_feats)
        loss_bg_fg = self.criterion[1](bg_feats, fg_feats)
        loss_fg_fg = self.criterion[2](fg_feats)
        loss_ccam = loss_bg_bg + loss_bg_fg + loss_fg_fg

        return ccam, loss_ccam


# -----Morphological Prototype Generator-----
class MorphologicalPrototypeGenerator(nn.Module):
    def __init__(
        self,
        num_classes : int,  # number of classes
        in_c : int,        # input channels
        # Patch Embed parameters
        patch_size : int,       # patch size
        embed_dim: int,  # embedding dimension
        # RoI Align parameters
        roi_out_size: Tuple[int, int],  # output size
        spatial_scale : float = 1/16,  # spatial scale
        sampling_ratio : int = 2,       # sampling ratio
        aligned : bool = True,            # aligned flag
    )->None:
        super(MorphologicalPrototypeGenerator, self).__init__()

        self.num_classes = num_classes
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.num_prototypes = num_classes
        self.patch_size = patch_size

        self.roi_align = RoIAlign(
            output_size=roi_out_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )

        self.ccam_generator = CCAMGenerator(in_c=in_c)

        self.patch_embed = PatchEmbed(
            img_size=roi_out_size,
            patch_size=patch_size,
            in_chans=in_c,
            embed_dim=embed_dim,
        )

        self.feature_transform = FeatureMapTransform(
            num_views=2,
            mask_prob=0.5,
            mask_scale=(0.02, 0.20),
            noise_prob=0.5,
            noise_sigma=0.05,
            keep_original=True,
        )

        self.gap_bg = nn.AdaptiveAvgPool2d((1, 1))


    def _get_patch_scores(
        self,
        ccams: torch.Tensor,    # [R * V, 1, H, W]
    )->torch.Tensor:
        """
        :return:
            patch_scores(torch.Tensor): fg and bg scores for each patch, [R * V, Np = num_patches, 2],
                                        patch_scores[:, :, 0] = fg_scores, patch_scores[:, :, 1] = bg_scores
        """
        if ccams.ndim != 4:
            raise ValueError(
                f"ccams 应为四维张量 [R * V, 1, H, W]，实际形状为 {tuple(ccams.shape)}"
            )

        _, channels, height, width = ccams.shape
        if channels != 1:
            raise ValueError(f"ccams 的通道数必须为 1，实际为 {channels}")

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                "CCAM 的空间尺寸必须能够被 patch_size 整除，"
                f"实际尺寸为 ({height}, {width})，patch_size={self.patch_size}"
            )

        # 使用与 PatchEmbed 相同的窗口和步长，计算每个 patch 的平均前景响应
        fg_scores = F.avg_pool2d(
            ccams,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).flatten(start_dim=1)  # [R * V, Np]

        bg_scores = 1.0 - fg_scores
        patch_scores = torch.stack(
            (fg_scores, bg_scores),
            dim=-1,
        )  # [R * V, Np, 2]

        return patch_scores


    def _get_anchor_features(
        self,
        patch_features : torch.Tensor,      # [Np, D]
        patch_fg_bg_scores : torch.Tensor,  # [Np, 2]
        lse_alpha: float = 10.0,
        topk_ratio: float = 0.2,
    )->torch.Tensor:
        """
        根据前景分数最高的 top-k patch 计算弱框的 anchor 特征。

        :param topk_ratio: 保留 patch 的比例，取值范围为 (0, 1]
        :return: anchor_feature，形状为 [D]
        """
        eps = 1e-6
        alpha = float(lse_alpha)

        if not 0.0 < topk_ratio <= 1.0:
            raise ValueError(f"topk_ratio 必须位于 (0, 1]，当前值为 {topk_ratio}")

        num_patches = patch_features.shape[0]
        if num_patches == 0:
            raise ValueError("patch_features 不能为空")
        if patch_fg_bg_scores.shape[0] != num_patches:
            raise ValueError("patch_features 与 patch_fg_bg_scores 的 patch 数量不一致")

        topk_count = max(1, math.ceil(num_patches * topk_ratio))
        all_fg_scores = patch_fg_bg_scores[:, 0]
        topk_indices = torch.topk(
            all_fg_scores,
            k=topk_count,
            largest=True,
            sorted=False,
        ).indices

        selected_features = patch_features.index_select(0, topk_indices)
        selected_fg_scores = all_fg_scores.index_select(0, topk_indices).clamp_min(eps)

        fg_weights = selected_fg_scores / (
            selected_fg_scores.sum(dim=0, keepdim=True) + eps
        )

        weighted_logits = alpha * selected_features + torch.log(fg_weights).unsqueeze(-1)
        anchor_feature = torch.logsumexp(weighted_logits, dim=0) / alpha
        # anchor_feature = F.normalize(anchor_feature, dim=-1, eps=eps)

        return anchor_feature


    def _get_morphological_prototypes(
        self,
        patch_features : torch.Tensor,      # [R, Np, D]
        weights : torch.Tensor,    # [R, Np]
        wb_labels : torch.Tensor,       # [R, num_classes]
        eps: float = 1e-6,
    )-> Dict[int, torch.Tensor]:
        """
        :return: prototypes, {class_id in [1, num_classes]: prototype tensor}
        """
        y_ = wb_labels[:, :, None, None]  # [R, num_classes, 1, 1]
        w_ = weights[:, None, :, None]  # [R, 1, Np, 1]
        f_ = patch_features[:, None, :, :]  # [R, 1, Np, D]

        numerator = (y_ * w_ * f_).sum(dim=(0, 2))  # sum over r and p => [num_classes, D]
        denom = (y_ * w_).sum(dim=(0, 2))  # [num_classes, 1]

        prototypes = numerator / (denom + eps)  # [num_classes, D]

        proto_dict : Dict[int, torch.Tensor] = {}
        for class_id in range(1, self.num_classes + 1):
            proto_dict[class_id] = prototypes[class_id - 1]

        return proto_dict


    def _get_background_prototype(
        self,
        patch_features : torch.Tensor,      # [R, Np, D]
        patch_fg_bg_scores : torch.Tensor,  # [R, Np, 2]
        top_k_ratio : float = 0.2,
        lse_alpha : float = 10.0,
    )-> torch.Tensor:
        '''
        Return:
            bg_prototype(torch.Tensor): background prototype, [D]
        '''
        # Get the top-k patches with the highest background scores
        k = int(patch_fg_bg_scores.shape[1] * top_k_ratio)
        _, top_k_indices = torch.topk(patch_fg_bg_scores[:, :, 1], k=k, dim=1)
        # Select the corresponding patch features
        bg_patch_features = patch_features.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, patch_features.shape[-1]))  # [R, k, D]
        R, k, D = bg_patch_features.shape
        bg_patch_features = bg_patch_features.reshape(R * k, D)  # [R * k, D]
        # Compute the LogSumExp of the selected background patch features
        bg_patch_features = lse_alpha * bg_patch_features  # [R * k, D]
        bg_prototype = torch.logsumexp(bg_patch_features, dim=0) / lse_alpha  # [D]

        return bg_prototype


    def forward(
        self,
        x : torch.Tensor,       # middle feature maps, [B, C2, H2, W2]
        wboxes : torch.Tensor,    # weak boxes, [R=num_wbs, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor,  # class label for weak boxes, [R, num_classes]
        lse_alpha : float = 10.0    # LSE alpha = 1 / tau, inverse temperature
    )-> Dict[str, Any]:
        """
        :return:
        out, Dict[str, Any], contains:
        - 'loss_ccam': torch.Tensor, CCAM loss
        - 'prototypes': Dict[int, torch.Tensor], {class_id in [0, num_classes](0 for background) : prototype_embeddings}
        - 'contrast_patch_features': torch.Tensor, patch features for SupCon, shape [R, V=num_views, D]
        """
        out : Dict[str, Any] = {}   # output
        # -----get aug weak box features & background features-----
        # aug weak box features
        roi_features = self.roi_align(x, wboxes)       # [R, C, H, W]
        aug_roi_features = self.feature_transform(roi_features)  # [R, V, C, H, W]


        # -----get CCAMs-----
        R, V, C, H, W = aug_roi_features.shape
        RV = R * V
        aug_roi_features = aug_roi_features.reshape(RV, C, H, W)    # [R * V, C, H, W]
        ccams, loss_ccam = self.ccam_generator(aug_roi_features)        # ccams of shape [R * V, 1, H, W]
        out.update({
            'loss_ccam' : loss_ccam,
            'ccams' : ccams,
        })


        # -----patch embed-----
        patch_features = self.patch_embed(
            aug_roi_features
        ).view(RV, -1, self.embed_dim)     # [R * V, Np=num_patches, D]
        D = patch_features.shape[2]


        # -----get contrast patch features for loss_constrain_supcon-----
        # LogSumExp for top-k patch features
        x_lse = lse_alpha * patch_features  # [R * V, Np, D]
        contrast_patch_features_lse = torch.logsumexp(x_lse, dim=1) / lse_alpha   # [R * V, D]
        # # mean pooling for contrast patch features
        # contrast_patch_features_mean = patch_features.mean(dim=1)  # [R * V, D]
        contrast_patch_features = contrast_patch_features_lse.view(R, V, D)     # for SupCon format, [R, V, D]
        out.update({
            'contrast_patch_features' : contrast_patch_features
        })


        # -----get morphological prototypes-----
        # get fg & bg scores for each patch
        patch_fg_bg_scores = self._get_patch_scores(ccams)      # [R * V, Np, 2], fg_scores = [:, :, 0], bg_scores = [:, :, 1]

        # # debug: record the patch scores
        # with open("debug/debug.txt", 'a') as f:
        #     f.write("patch_fg_bg_scores:\n")
        #     for idx, s in enumerate(patch_fg_bg_scores):
        #         f.write(f"wbox {idx}:\n")
        #         for s_i in s:
        #             f.write(f"fg: {s_i[0]}, bg: {s_i[1]}\n")
        #         f.write("\n")

        # build background prototype
        bg_prototype = self._get_background_prototype(patch_features, patch_fg_bg_scores)  # [D]
        # get anchor feature of each weak box
        anchor_features_list : List[torch.Tensor] = []
        for i in range(RV):
            patch_feature = patch_features[i]
            patch_fg_bg_score = patch_fg_bg_scores[i]
            anchor_feature = self._get_anchor_features(patch_feature, patch_fg_bg_score, lse_alpha=lse_alpha)
            anchor_features_list.append(anchor_feature)
        anchor_features = torch.stack(anchor_features_list, dim=0)      # [R * V, D]
        # compute patch weights(similarity)
        patch_features_norm = F.normalize(patch_features, dim=-1)
        anchor_features_norm = F.normalize(anchor_features, dim=-1)
        weights = (patch_features_norm * anchor_features_norm.unsqueeze(1)).sum(dim=-1)  # [R * V, Np]
        weights = F.relu(weights)  # remove negative value
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # normalize, sum to 1
        # prototypes: Dict[int, torch.Tensor], {class_id in [1, num_classes] : prototype tensor}
        expand_wb_labels = wb_labels.unsqueeze(1).expand(-1, V, -1).reshape(RV, -1)  # [R * V, num_classes]
        prototypes = self._get_morphological_prototypes(patch_features, weights, expand_wb_labels)
        prototypes = {
            0: bg_prototype,
            **prototypes,
        }      # 0 for background
        
        out.update({
            'prototypes' : prototypes
        })

        return out


# -----Morphological Prototype Builder-----
class ProtypeBuilder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,  # Default VGG-16
        hook: FeatureHook,         # Feature Hook
        mp_generator: MorphologicalPrototypeGenerator,  # Morphological Prototype Generator
        cfg : PrototypeBuilderConfig       # configuration
    )-> None:
        super(ProtypeBuilder, self).__init__()

        self.encoder = backbone
        self.hook = hook
        self.mp_generator = mp_generator
        self.projector = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim)
        )
        self.cfg = cfg

        self.apply(self._init_weights)


    def _init_weights(self, m)->None:
        """
        Initialize weights for Linear and BatchNorm layers.
        :param m: Module to initialize
        """
        if isinstance(m, nn.Linear):  # Check if the module is a Linear layer
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
            if m.bias is not None:  # Initialize bias to zero if it exists
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):  # Check if the module is a BatchNorm layer
            nn.init.constant_(m.weight, 1.0)  # Initialize scale (gamma) to 1
            nn.init.constant_(m.bias, 0)  # Initialize shift (beta) to 0


    def eval_prototype(
        self,
        x: torch.Tensor,  # input images, [B, C, H, W]
        boxes: torch.Tensor,  # 待评估框，[R, 5]，每行为 [batch_idx, x1, y1, x2, y2]
        boxes_labels: torch.Tensor,  # 待评估框的类别标签，[R, num_classes]
        prototypes: Dict[int, torch.Tensor],  # raw, {class_id, prototype tensor}
        return_details: bool = True,
        lse_alpha: float = 10.0,
        lse_eps: float = 1e-6,
        positive_only: bool = False,
    ) -> Dict[str, Any]:
        """评估框特征与对应类别原型的相似度。

        positive_only=True 时计算每个框与所有类别原型的 positive similarity，
        不计算负类相似度、margin 等其他指标，用于背景框评估。
        """
        self.hook.clear()
        _ = self.encoder(x)
        feature_maps = self.hook.outputs
        mid_feature_maps = feature_maps['mid']
        roi_features = self.mp_generator.roi_align(mid_feature_maps, boxes)  # [R, C, H, W]
        patch_features = self.mp_generator.patch_embed(roi_features)  # [R, Np=num_patches, D]
        patch_features = self.projector(patch_features)  # [R, Np, D]

        # sort by class_id
        class_ids = sorted(list(prototypes.keys()))
        # norm
        proto_list = [prototypes[k] for k in class_ids]
        proto_norm_list = []
        for proto in proto_list:
            proto_norm = F.normalize(proto, dim=-1)
            proto_norm_list.append(proto_norm)
        proto_mat = torch.stack(proto_norm_list, dim=0)  # [num_classes + 1, D]

        R, num_classes = boxes_labels.shape
        sims_sum = {k: 0.0 for k in range(num_classes)}
        sims_cnt = {k: 0 for k in range(num_classes)}

        details = {
            "pos": {k: [] for k in range(num_classes)},
        }
        if not positive_only:
            details.update({
                "neg_max": {k: [] for k in range(num_classes)},  # 最大负类相似度
                "margin": {k: [] for k in range(num_classes)},  # 正类与最大负类的间隔
                "bg": {k: [] for k in range(num_classes)},  # 与背景原型的相似度
            })

        for i in range(R):
            box_patch = patch_features[i]  # [num_patches, D]

            # # mean pooling
            # box_vec = box_patch.mean(dim=0, keepdim=True)  # [1, D]
            # box_vec = F.normalize(box_vec, dim=1)  # [1, D]

            # LogSumExp pooling
            box_patch = box_patch * lse_alpha  # [num_patches, D]
            box_lse = torch.logsumexp(box_patch, dim=0, keepdim=True) / lse_alpha
            box_vec = F.normalize(box_lse, dim=1, eps=lse_eps)  # [1, D]

            if positive_only:
                sims_all = torch.matmul(box_vec, proto_mat.t()).squeeze(0)
                for class_id in range(num_classes):
                    sim_pos = sims_all[class_id].item()
                    sims_sum[class_id] += sim_pos
                    sims_cnt[class_id] += 1
                    if return_details:
                        details["pos"][class_id].append(sim_pos)
                continue

            box_label = torch.argmax(boxes_labels[i]).item()
            sims_all = torch.matmul(box_vec, proto_mat.t()).squeeze(0)  # [num_classes]
            sim_pos = sims_all[box_label].item()
            sims_sum[box_label] += sim_pos
            sims_cnt[box_label] += 1
            if return_details:
                details["pos"][box_label].append(sim_pos)

            sim_bg = sims_all[0].item() if 0 in class_ids else float("nan")

            # max negative class similarity
            if num_classes > 1:
                mask = torch.ones(num_classes, dtype=torch.bool, device=sims_all.device)
                mask[box_label] = False
                sim_neg_max = sims_all[mask].max().item()
            else:
                sim_neg_max = float("-inf")

            margin = sim_pos - sim_neg_max if sim_neg_max != float("-inf") else float("inf")

            if return_details:
                details["neg_max"][box_label].append(sim_neg_max)
                details["margin"][box_label].append(margin)
                details["bg"][box_label].append(sim_bg)

        out = {
            "sum": sims_sum,
            "cnt": sims_cnt,
        }
        if return_details:
            out["details"] = details

        return out


    def get_feature_maps(
        self,
        x: torch.Tensor,        # input images, [B, C, H, W]
    )-> Dict[str, Any]:
        '''
        Return:
            self.hook.outputs(Dict[str, Any]): a dict containing feature maps from different layers of the backbone, keys:
            - 'low'(torch.Tensor): low-level feature maps, shape [B, C1, H1, W1]
            - 'mid'(torch.Tensor): mid-level feature maps, shape [B, C2, H2, W2]
            - 'high'(torch.Tensor): high-level feature maps, shape [B, C3, H3, W3]
        '''
        self.hook.clear()
        _ = self.encoder(x)
        
        return self.hook.outputs


    def forward(
        self,
        x: torch.Tensor,        # input images, [B, C, H, W]
        wboxes: torch.Tensor,  # weak boxes, [R=num_wbs, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels: torch.Tensor,  # class label for weak boxes, [R, num_classes]
    )-> Dict[str, Any]:
        """
        :return:
        out: Dict[str, Any], contains:
        - 'loss_ccam': Tensor, CCAM loss
        - 'ccams': Tensor, CCAM 概率图，形状为 [R * V, 1, H, W]
        - 'prototypes': Dict[int, torch.Tensor], {class_id in [0, num_classes](0 for background) : prototype_embeddings}
        - 'contrast_patch_features': torch.Tensor, patch features for SupCon, shape [R, V=num_views, D]
        """
        out : Dict[str, Any] = {}   # output

        # -----get multi-level feature maps-----
        self.hook.clear()
        _ = self.encoder(x)
        feature_maps = self.hook.outputs
        mid_feature_maps = feature_maps['mid']      # [B, C2, H2, W2]

        # -----construct prototypes-----
        # prototypes: Dict[int, Tensor], {class_id, prototype_embeddings_raw}
        mp_g_out = self.mp_generator(
            mid_feature_maps,
            wboxes,
            wb_labels
        )

        # -----projection-----
        contrast_patch_features = mp_g_out['contrast_patch_features']       # [R, V, D]
        R, V, D = contrast_patch_features.shape
        contrast_patch_features = contrast_patch_features.reshape(R * V, D)     # [R * V, D]
        contrast_patch_features = self.projector(contrast_patch_features).view(R, V, D)     # [R, V, D]
        prototypes_dict = mp_g_out['prototypes']  # {class_id, prototype_embeddings_raw}
        proto_keys = sorted(prototypes_dict.keys())
        expected_proto_keys = list(range(self.mp_generator.num_classes + 1))
        if proto_keys != expected_proto_keys:
            raise ValueError(f"原型 key 应为 {expected_proto_keys}，实际为 {proto_keys}。")
        prototypes = torch.stack([prototypes_dict[k] for k in proto_keys], dim=0)     # [num_classes + 1, D]
        prototypes = self.projector(prototypes)     # [num_classes + 1, D]
        for idx, key in enumerate(proto_keys):
            prototypes_dict[key] = prototypes[idx]

        out.update({
            'loss_ccam' : mp_g_out['loss_ccam'],
            'ccams' : mp_g_out['ccams'],
            'prototypes' : prototypes_dict,
            'contrast_patch_features' : contrast_patch_features,
        })

        return out


def build_prototype_builder_model(
    cfg: Dict      # global configuration
)-> ProtypeBuilder:
    # init backbone and hook
    backbone, hook = build_vgg16_backbone_with_hook(cfg['MODEL']['LAYER_INDICES'])

    # build model config
    model_cfg = build_prototype_builder_config(cfg)

    # init mp_generator
    mp_generator = MorphologicalPrototypeGenerator(
        num_classes=cfg['DATA']['NUM_CLASSES'],
        in_c=model_cfg.in_c,
        patch_size=model_cfg.patch_size,
        embed_dim=model_cfg.embed_dim,
        roi_out_size=model_cfg.roi_out_size_mid,
        spatial_scale=model_cfg.spatial_scale_mid,
        sampling_ratio=model_cfg.sampling_ratio
    )

    return ProtypeBuilder(backbone, hook, mp_generator, model_cfg)
