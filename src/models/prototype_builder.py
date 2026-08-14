# Prototype Builder, only construct morphological prototypes
from collections import OrderedDict
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


# -----Class Activation Map Head-----
# generate CAMs and compute CAM loss
class CAMHead(nn.Module):
    def __init__(
        self,
        num_classes: int, # number of classes
        in_channels: int  # input channels
    )-> None:
        super(CAMHead, self).__init__()

        self.num_classes = num_classes

        self.CE_loss = nn.CrossEntropyLoss()
        self.cam_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

        weight_init.c2_msra_fill(self.cam_conv)

    def forward(self, x : torch.Tensor, y : torch.Tensor)-> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param x: RoI feature maps, [R, C, H, W]
        :param y: weak box labels, [R, num_classes]
        :return:
        - CAMs, [R, num_classes, H, W]
        - CAM loss, {loss_name: loss}
        """
        # get CAMs
        x = self.cam_conv(x)

        # get class logits
        logits = F.avg_pool2d(x, (x.size(2), x.size(3)))
        logits = logits.view(-1, self.num_classes)

        # compute CE Loss
        target = torch.argmax(y, dim=1)
        loss_cam = self.CE_loss(logits, target)

        return x, loss_cam


# -----Feature Augmentation Transform-----
class FeatureMapTransform(nn.Module):
    def __init__(
        self,
        num_views: int = 2,
        mask_prob: float = 0.5,
        mask_scale: Tuple[float, float] = (0.02, 0.20),
        noise_prob: float = 0.5,
        noise_sigma: float = 0.05,
        keep_original: bool = True,
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

        self.cam_head = CAMHead(
            num_classes=num_classes,
            in_channels=in_c,
        )

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


    def _cam_to_patch_fg_bg_scores(self, cams : torch.Tensor,wb_labels : torch.Tensor)-> torch.Tensor:
        """
        :param cams: CAMs, [R, K = num_classes, H, W]
        :param wb_labels: weak box labels, [R, K = num_classes]
        :return: patch_fg_bg_scores : fg/bg scores for each patch, [R, Np = num_patches, 2]
        """

        R, K, H, W = cams.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, "RoI output size must be divisible by patch_size"

        # get class ids for each weak box
        cls_ids = torch.argmax(wb_labels, dim=1)  # [R]

        # gather class-specific CAM: [R, H, W]
        cam_cls = cams[torch.arange(R, device=cams.device), cls_ids]  # [R, H, W]

        # get probability maps from CAMs
        cam_prob = torch.sigmoid(cam_cls).unsqueeze(1)  # [R, 1, H, W]

        fg_map = F.avg_pool2d(
            cam_prob,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )       # [R, 1, Hp, Wp]
        fg_score = fg_map.flatten(1)        # [R, N]
        bg_score = 1 - fg_score     # [R, N]

        patch_fg_bg_scores = torch.stack([fg_score, bg_score], dim=-1)  # [R, Np, 2]

        return patch_fg_bg_scores


    def _get_anchor_features(
        self,
        patch_features : torch.Tensor,      # [Np, D]
        patch_fg_bg_scores : torch.Tensor,  # [Np, 2]
        lse_alpha: float = 10.0,
    )->torch.Tensor:
        """
        :return: anchor_feature: the anchor feature of weak box, [D]
        """
        eps = 1e-6
        alpha = float(lse_alpha)

        patch_features_norm = F.normalize(patch_features, dim=-1, eps=eps)

        fg_scores = patch_fg_bg_scores[:, 0].clamp_min(eps)
        fg_weights = fg_scores / (fg_scores.sum(dim=0, keepdim=True) + eps)

        weighted_logits = alpha * patch_features_norm + torch.log(fg_weights).unsqueeze(-1)
        anchor_feature = torch.logsumexp(weighted_logits, dim=0) / alpha
        anchor_feature = F.normalize(anchor_feature, dim=-1, eps=eps)

        return anchor_feature


    def _get_morphological_prototypes(
        self,
        patch_features : torch.Tensor,      # [R, Np, D]
        weights : torch.Tensor,    # [R, Np]
        wb_labels : torch.Tensor,       # [R, num_classes]
        eps: float = 1e-6,
        normalize_proto: bool = False
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
        if normalize_proto:
            prototypes = F.normalize(prototypes, dim=-1)

        proto_dict : Dict[int, torch.Tensor] = {}
        for class_id in range(1, self.num_classes + 1):
            proto_dict[class_id] = prototypes[class_id - 1]

        return proto_dict


    def _get_background_prototype(
        self,
        patch_features : torch.Tensor,      # [R, Np, D]
        patch_fg_bg_scores : torch.Tensor,  # [R, Np, 2]
        top_k_ratio : float = 0.05,
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
        # Compute the mean of the selected background patch features
        bg_prototype = bg_patch_features.mean(dim=0)  # [D]

        return bg_prototype


    def forward(
        self,
        x : torch.Tensor,       # middle feature maps, [B, C2, H2, W2]
        wboxes : torch.Tensor,    # weak boxes, [R=num_wbs, 5], for each box, [batch_idx, x1, y1, x2, y2]
        wb_labels : torch.Tensor,    # class label for weak boxes, [R, num_classes]
        lse_alpha : float = 10.0    # LSE alpha = 1 / tau, inverse temperature
    )-> Dict[str, Any]:
        """
        :return:
        out, Dict[str, Any], contains:
        - 'loss_cam': torch.Tensor, CAM loss
        - 'prototypes': Dict[int, torch.Tensor], {class_id in [0, num_classes](0 for background) : prototype_embeddings}
        - 'contrast_patch_features': torch.Tensor, patch features for SupCon, shape [R, V=num_views, D]
        """
        out : Dict[str, Any] = {}   # output
        # -----get aug weak box features & background features-----
        # aug weak box features
        roi_features = self.roi_align(x, wboxes)       # [R, C, H, W]
        # aug_roi_features = self.feature_transform(roi_features.detach())  # [R, V, C, H, W]
        aug_roi_features = self.feature_transform(roi_features)  # [R, V, C, H, W]


        # -----get CAMs-----
        R, V, C, H, W = aug_roi_features.shape
        RV = R * V
        expand_wb_labels = wb_labels.unsqueeze(1).expand(-1, V, -1).reshape(RV, -1)  # [R * V, num_classes]
        aug_roi_features = aug_roi_features.reshape(RV, C, H, W)    # [R * V, C, H, W]
        cams, loss_cam = self.cam_head(aug_roi_features, expand_wb_labels)        # cams, [R * V, num_classes, H, W]
        out.update({
            'loss_cam' : loss_cam,
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
        patch_fg_bg_scores = self._cam_to_patch_fg_bg_scores(cams, expand_wb_labels)      # [R * V, Np, 2], fg_scores = [:, :, 0], bg_scores = [:, :, 1]
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
        prototypes = self._get_morphological_prototypes(patch_features_norm, weights, expand_wb_labels)
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
        boxes: torch.Tensor,  # GT boxes for RoI Align, [R=num_gt_boxes, 5], for each box, [batch_idx, x1, y1, x2, y2]
        boxes_labels: torch.Tensor,  # class labels for GT boxes, [R, num_classes]
        prototypes: Dict[int, torch.Tensor],  # raw, {class_id, prototype tensor}
        return_details: bool = True,
        lse_alpha: float = 10.0,
        lse_eps: float = 1e-6
    ) -> Dict[str, Any]:
        '''
        :return: Similarity of GT and prototypes, {class_id : average_similarity}
        '''
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

        # statistics: results details
        details = {
            "pos": {k: [] for k in range(num_classes)},  # sim_pos of each GT
            "neg_max": {k: [] for k in range(num_classes)},  # max sim_neg of each GT
            "margin": {k: [] for k in range(num_classes)},  # margin = sim_pos - max sim_neg
            "bg": {k: [] for k in range(num_classes)},  # 每个 GT box 与背景原型的相似度
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

            sim_pos = sims_all[gt_label].item()  # positive class similarity
            sim_bg = sims_all[0].item() if 0 in class_ids else float("nan")

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
                details["bg"][gt_label].append(sim_bg)

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
        - 'loss_cam': Tensor, CAM loss
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
            'loss_cam' : mp_g_out['loss_cam'],
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