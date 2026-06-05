# Other Loss Functions
import torch
import torch.nn.functional as F


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