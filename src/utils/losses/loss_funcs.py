# Other Loss Functions
import torch
import torch.nn.functional as F
from typing import Dict

from .supcon_loss import SupConLoss


# -----Patch CLS Loss-----
def get_patch_cls_loss(
    patch_logits : torch.Tensor,  # [R * k, num_classes], R=num_wbs, k=num_patches_per_wb
    wb_labels : torch.Tensor,      # [R, num_classes]
)->torch.Tensor:
    '''return loss_patch'''
    k = patch_logits.shape[0] // wb_labels.shape[0]  # number of patches per weak box
    targets = wb_labels.argmax(dim=1).repeat_interleave(k)
    loss_patch_cls = F.cross_entropy(patch_logits, targets)

    return loss_patch_cls


# -----Prototypes Loss-----
def get_proto_loss(
    contrast_patch_features: torch.Tensor,      # norm, shape [R=num_wbs, V=num_views, D]
    all_labels: torch.Tensor,                 # range in [0, num_classes-1], shape [R=num_wbs]
    prototypes: Dict[int, torch.Tensor],    # {class_id in [0, num_classes](0 for background) : prototype_embeddings_norm}
    tau: float = 0.07
)-> torch.Tensor:
    '''return loss_proto'''
    R, V, D = contrast_patch_features.shape
    z = contrast_patch_features.view(R * V, D)     # [R * V, D]
    proto_keys = sorted(prototypes.keys())
    expected_proto_keys = list(range(len(proto_keys)))
    if proto_keys != expected_proto_keys:
        raise ValueError(f"原型 key 应为连续编号 {expected_proto_keys}，实际为 {proto_keys}。")
    p = torch.stack([prototypes[k] for k in proto_keys], dim=0)  # [num_classes + 1, D]

    logits = (z @ p.t()) / tau      # [R * V, num_classes + 1]
    log_prob = F.log_softmax(logits, dim=1)    # [R * V, num_classes + 1]
    targets = all_labels.unsqueeze(1).expand(-1, V).reshape(R * V) + 1     # 0 for background, [1, num_classes] for foreground classes
    log_p_true = log_prob.gather(1, targets.view(-1, 1)).squeeze(1)  # [R * V]
    loss_proto = -log_p_true.mean()

    return loss_proto


# -----Constrain Loss-----
def get_constrain_loss(
    contrast_patch_features: torch.Tensor,      # norm, shape [R=num_wbs, V=num_views, D]
    all_labels: torch.Tensor,                 # range in [0, num_classes-1], shape [R=num_wbs]
    prototypes: Dict[int, torch.Tensor],    # {class_id in [0, num_classes](0 for background) : prototype_embeddings_norm}
    omega_1: float = 0.6,
    omega_2: float = 0.4,
)-> Dict[str, torch.Tensor]:
    '''
    :return:
    loss_constrain_dict: Dict[str, torch.Tensor], constrains:
    - 'loss_constrain': total constrain loss
    - 'loss_constrain_supcon': SupCon loss component
    - 'loss_constrain_proto': Proto loss component
    '''
    # -----compute SupCon loss-----
    supcon_loss_fn = SupConLoss()
    loss_constrain_supcon = supcon_loss_fn(contrast_patch_features, all_labels)

    # -----compute Proto loss-----
    loss_constrain_proto = get_proto_loss(contrast_patch_features, all_labels, prototypes)

    # -----weighted sum-----
    loss_constrain = omega_1 * loss_constrain_supcon + omega_2 * loss_constrain_proto

    loss_constrain_dict = {
        'loss_constrain': loss_constrain,
        'loss_constrain_supcon': loss_constrain_supcon,
        'loss_constrain_proto': loss_constrain_proto
    }

    return loss_constrain_dict


# -----Sep Loss-----
def get_sep_loss(
    dataset_prototypes: Dict[int, torch.Tensor],    # {class_id in [0, num_classes](0 for background) : prototype_embeddings_norm_detached}
    batch_prototypes: Dict[int, torch.Tensor],    # {class_id in [0, num_classes](0 for background) : prototype_embeddings_norm}
    omega_bg: float = 1.0,      # bg scale factor
    margin_fg: float = 0.2,
    margin_bg: float = 0.0,
    valid_fg_class_ids: torch.Tensor | None = None,
)-> Dict[str, torch.Tensor]:
    '''
    :return: loss_sep_dict(Dict[str, torch.Tensor]): contains:
    - 'loss_sep'(torch.Tensor): total separation loss
    - 'loss_sep_fg'(torch.Tensor): foreground class separation loss
    - 'loss_sep_bg'(torch.Tensor): background class separation loss
    '''
    if len(batch_prototypes) == 0:
        raise ValueError("batch_prototypes must not be empty.")
    if len(dataset_prototypes) == 0:
        raise ValueError("dataset_prototypes must not be empty.")

    proto_keys = sorted(batch_prototypes.keys())
    dataset_proto_keys = sorted(dataset_prototypes.keys())
    expected_proto_keys = list(range(len(proto_keys)))
    if proto_keys != expected_proto_keys:
        raise ValueError(f"batch prototype keys must be {expected_proto_keys}, got {proto_keys}.")
    if dataset_proto_keys != proto_keys:
        raise ValueError(f"dataset prototype keys must be {proto_keys}, got {dataset_proto_keys}.")
    if 0 not in proto_keys:
        raise ValueError("prototype key 0 is required for background.")

    ref_proto = next(iter(batch_prototypes.values()))
    device = ref_proto.device
    dtype = ref_proto.dtype
    zero = ref_proto.sum() * 0.0

    for key in proto_keys:
        if batch_prototypes[key].device != device or dataset_prototypes[key].device != device:
            raise ValueError(f"prototype device mismatch at key {key}.")
        if batch_prototypes[key].dtype != dtype or dataset_prototypes[key].dtype != dtype:
            raise ValueError(f"prototype dtype mismatch at key {key}.")

    fg_keys = [k for k in proto_keys if k != 0]
    if valid_fg_class_ids is not None:
        valid_keys = sorted({int(k) for k in valid_fg_class_ids.detach().cpu().tolist()})
        fg_keys = [k for k in valid_keys if k != 0]
        missing_keys = sorted(set(fg_keys) - set(proto_keys))
        if missing_keys:
            raise ValueError(f"valid foreground prototype keys are missing: {missing_keys}.")

    if len(fg_keys) == 0:
        return {
            'loss_sep': zero,
            'loss_sep_fg': zero,
            'loss_sep_bg': zero,
        }

    batch_fg = torch.stack([batch_prototypes[k] for k in fg_keys], dim=0)
    dataset_fg_keys = [k for k in proto_keys if k != 0]
    dataset_fg = torch.stack([dataset_prototypes[k] for k in dataset_fg_keys], dim=0)

    sim_fg = batch_fg @ dataset_fg.t()
    fg_pair_mask = torch.tensor(
        [[i != j for j in dataset_fg_keys] for i in fg_keys],
        dtype=torch.bool,
        device=device,
    )
    if fg_pair_mask.any():
        loss_sep_fg = torch.relu(sim_fg - margin_fg).pow(2)[fg_pair_mask].mean()
    else:
        loss_sep_fg = zero

    dataset_bg = dataset_prototypes[0]
    sim_bg = batch_fg @ dataset_bg
    loss_sep_bg = torch.relu(sim_bg - margin_bg).pow(2).mean()

    loss_sep = loss_sep_fg + omega_bg * loss_sep_bg
    loss_sep_dict = {
        'loss_sep': loss_sep,
        'loss_sep_fg': loss_sep_fg,
        'loss_sep_bg': loss_sep_bg,
    }

    return loss_sep_dict