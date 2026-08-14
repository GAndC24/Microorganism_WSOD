"将弱框对应的类别激活图投影到原图并保存。"

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from .vis_boxes import _clip_box, _draw_labeled_box, _read_gt_boxes, _resolve_sample_paths


_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / 'debug' / 'vis_cams'
_GT_COLOR = (40, 210, 80)
_WEAK_BOX_COLOR = (0, 220, 255)
_CAM_ALPHA = 0.6


def _collect_target_boxes(
    targets: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, list[int]]:
    """按 batch 顺序展开 ``targets[*]['boxes']``。"""
    box_batches: list[torch.Tensor] = []
    batch_indices: list[int] = []
    for batch_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise TypeError(f"targets[{batch_index}] 必须是映射类型")
        boxes = target.get("boxes")
        if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"targets[{batch_index}]['boxes'] 必须是形状为 [N, 4] 的 Tensor")

        boxes_cpu = boxes.detach().float().cpu()
        if not torch.isfinite(boxes_cpu).all():
            raise ValueError(f"targets[{batch_index}]['boxes'] 包含非有限坐标")
        box_batches.append(boxes_cpu)
        batch_indices.extend([batch_index] * int(boxes_cpu.shape[0]))

    if not box_batches:
        return torch.empty((0, 4), dtype=torch.float32), []
    return torch.cat(box_batches, dim=0), batch_indices


def _validate_inputs(
    cams: torch.Tensor,
    num_weak_boxes: int,
    wb_one_hot_labels: torch.Tensor,
) -> tuple[int, int]:
    """校验输入，并返回弱框数和每个弱框的 CAM 视图数。"""
    if not isinstance(cams, torch.Tensor) or cams.ndim != 4:
        raise ValueError("cams 必须是形状为 [R * V, K, H, W] 的 Tensor")
    if not isinstance(wb_one_hot_labels, torch.Tensor) or wb_one_hot_labels.ndim != 2:
        raise ValueError("wb_one_hot_labels 必须是形状为 [R, K] 的 Tensor")

    if wb_one_hot_labels.shape[0] != num_weak_boxes:
        raise ValueError("targets 中的 boxes 与 wb_one_hot_labels 的弱框数量不一致")
    if cams.shape[1] != wb_one_hot_labels.shape[1]:
        raise ValueError("cams 与 wb_one_hot_labels 的类别数量不一致")
    if num_weak_boxes == 0:
        return 0, 0
    if cams.shape[0] % num_weak_boxes != 0:
        raise ValueError("cams 的首维必须是弱框数量 R 的整数倍")
    if cams.shape[2] <= 0 or cams.shape[3] <= 0:
        raise ValueError("CAM 的空间尺寸必须大于 0")

    return num_weak_boxes, int(cams.shape[0] // num_weak_boxes)


def _validate_canvas_sizes(
    canvas_sizes: Sequence[tuple[int, int]],
    num_targets: int,
) -> list[tuple[int, int]]:
    """校验并规范化每张 batch 图像的 ``(高, 宽)``。"""
    if len(canvas_sizes) != num_targets:
        raise ValueError("canvas_sizes 与 targets 的图片数量不一致")

    validated_sizes: list[tuple[int, int]] = []
    for batch_index, canvas_size in enumerate(canvas_sizes):
        if len(canvas_size) != 2:
            raise ValueError(f"canvas_sizes[{batch_index}] 必须包含高和宽")
        canvas_height, canvas_width = (int(value) for value in canvas_size)
        if canvas_height <= 0 or canvas_width <= 0:
            raise ValueError(f"canvas_sizes[{batch_index}] 的高和宽必须大于 0")
        validated_sizes.append((canvas_height, canvas_width))
    return validated_sizes


def _prepare_visualization_canvas(
    original_image: Image.Image,
    gt_boxes: Sequence[tuple[str, tuple[float, float, float, float]]],
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, list[tuple[str, tuple[float, float, float, float]]]]:
    """将原图和 GT 框转换到弱框所使用的训练画布坐标系。"""
    canvas_height, canvas_width = canvas_size
    original_width, original_height = original_image.size
    scale_x = canvas_width / original_width
    scale_y = canvas_height / original_height
    scaled_gt_boxes = [
        (
            class_name,
            (
                box[0] * scale_x,
                box[1] * scale_y,
                box[2] * scale_x,
                box[3] * scale_y,
            ),
        )
        for class_name, box in gt_boxes
    ]
    image = original_image.resize(
        (canvas_width, canvas_height),
        Image.Resampling.BILINEAR,
    )
    return image, scaled_gt_boxes


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    """将 CAM logits 转为稳定的 [0, 1] 可视化强度。"""
    probability = torch.sigmoid(cam.detach().float().cpu())
    probability = torch.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    minimum = probability.min()
    maximum = probability.max()
    if float(maximum - minimum) <= torch.finfo(probability.dtype).eps:
        return torch.zeros_like(probability)
    return ((probability - minimum) / (maximum - minimum)).clamp(0.0, 1.0)


def _cam_to_heatmap(cam: torch.Tensor) -> Image.Image:
    """使用无额外依赖的 Jet 色表将单通道 CAM 转为 RGB 图。"""
    value = cam.clamp(0.0, 1.0)
    red = (1.5 - torch.abs(4.0 * value - 3.0)).clamp(0.0, 1.0)
    green = (1.5 - torch.abs(4.0 * value - 2.0)).clamp(0.0, 1.0)
    blue = (1.5 - torch.abs(4.0 * value - 1.0)).clamp(0.0, 1.0)
    rgb = torch.stack((red, green, blue), dim=-1).mul(255).byte().numpy()
    return Image.fromarray(rgb, mode="RGB")


def _overlay_cam(
    image: Image.Image,
    cam: torch.Tensor,
    weak_box: tuple[float, float, float, float],
) -> None:
    """仅在弱框内部叠加 CAM，避免把 ROI 激活错误扩散到整张图。"""
    image_width, image_height = image.size
    clipped_box = _clip_box(weak_box, image_width, image_height)
    if clipped_box is None:
        return

    x1, y1, x2, y2 = clipped_box
    left = max(0, math.floor(x1))
    top = max(0, math.floor(y1))
    right = min(image_width, math.ceil(x2))
    bottom = min(image_height, math.ceil(y2))
    if right <= left or bottom <= top:
        return

    region_size = (right - left, bottom - top)
    normalized_cam = _normalize_cam(cam)
    heatmap = _cam_to_heatmap(normalized_cam).resize(region_size, Image.Resampling.BILINEAR)
    alpha = Image.fromarray(
        normalized_cam.mul(255 * _CAM_ALPHA).byte().numpy(),
        mode="L",
    ).resize(region_size, Image.Resampling.BILINEAR)
    image_region = image.crop((left, top, right, bottom))
    image.paste(Image.composite(heatmap, image_region, alpha), (left, top))


def _draw_annotations(
    image: Image.Image,
    gt_boxes: Sequence[tuple[str, tuple[float, float, float, float]]],
    weak_boxes: Sequence[
        tuple[int, tuple[float, float, float, float], int]
    ],
) -> None:
    """在 CAM 之上绘制全部 GT 框、弱框和图例。"""
    image_width, image_height = image.size
    line_width = max(2, round(min(image_width, image_height) / 250))
    draw = ImageDraw.Draw(image)

    for class_name, gt_box in gt_boxes:
        clipped_gt_box = _clip_box(gt_box, image_width, image_height)
        if clipped_gt_box is not None:
            _draw_labeled_box(
                draw,
                clipped_gt_box,
                _GT_COLOR,
                line_width,
                label=f"GT: {class_name}",
            )

    for weak_box_index, weak_box, class_id in weak_boxes:
        clipped_weak_box = _clip_box(weak_box, image_width, image_height)
        if clipped_weak_box is not None:
            _draw_labeled_box(
                draw,
                clipped_weak_box,
                _WEAK_BOX_COLOR,
                line_width,
                label=f"Weak {weak_box_index}: class {class_id}",
            )

    legend = "CAM    Weak box    GT"
    text_box = draw.textbbox((6, 6), legend)
    draw.rectangle((3, 3, text_box[2] + 9, text_box[3] + 9), fill=(0, 0, 0))
    draw.text((6, 6), "CAM", fill=(255, 80, 40))
    weak_x = 6 + draw.textlength("CAM    ")
    draw.text((weak_x, 6), "Weak box", fill=_WEAK_BOX_COLOR)
    gt_x = 6 + draw.textlength("CAM    Weak box    ")
    draw.text((gt_x, 6), "GT", fill=_GT_COLOR)


def visualize_cams(
    cams: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    wb_one_hot_labels: torch.Tensor,
    canvas_sizes: Sequence[tuple[int, int]],
    epoch: int,
    iter: int,
) -> list[Path]:
    """
    按图片可视化一个 batch 内全部弱框的类别 CAM。

    ``cams`` 可以是 ``[R, K, H, W]``，也可以是包含增强视图的
    ``[R * V, K, H, W]``。后者按模型的排列规则还原成 ``[R, V, ...]``，
    并使用第 0 个未增强视图。每个输出文件对应一张图片，包含训练画布图像、
    全部 GT 框、全部弱框，以及投影在各自弱框内部的类别 CAM。

    弱框直接取自 ``targets[*]['boxes']``，展开顺序与训练时构造标签的顺序一致。
    ``canvas_sizes`` 必须按 batch 顺序显式提供每张训练图像的 ``(高, 宽)``。
    """
    validated_canvas_sizes = _validate_canvas_sizes(canvas_sizes, len(targets))
    boxes_cpu, box_batch_indices = _collect_target_boxes(targets)
    num_weak_boxes, num_views = _validate_inputs(
        cams,
        int(boxes_cpu.shape[0]),
        wb_one_hot_labels,
    )
    if num_weak_boxes == 0:
        return []

    try:
        epoch_index = int(epoch)
        iter_index = int(iter)
    except (TypeError, ValueError) as exc:
        raise ValueError("epoch 和 iter 必须是整数") from exc
    if epoch_index < 0 or iter_index < 0:
        raise ValueError("epoch 和 iter 不能为负数")

    cams_by_box = cams.detach().reshape(
        num_weak_boxes,
        num_views,
        cams.shape[1],
        cams.shape[2],
        cams.shape[3],
    )[:, 0].cpu()
    class_ids = wb_one_hot_labels.detach().argmax(dim=1).cpu()

    output_dir = _OUTPUT_ROOT / f"epoch_{epoch_index:03d}" / f"iter_{iter_index:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_cache: dict[
        str,
        tuple[Image.Image, list[tuple[str, tuple[float, float, float, float]]]],
    ] = {}
    output_paths: list[Path] = []

    weak_box_indices_by_batch: dict[int, list[int]] = {}
    for weak_box_index, batch_index in enumerate(box_batch_indices):
        weak_box_indices_by_batch.setdefault(batch_index, []).append(weak_box_index)

    for batch_index, weak_box_indices in weak_box_indices_by_batch.items():
        target = targets[batch_index]
        image_id = str(target.get("image_id", "")).strip()
        if not image_id:
            raise ValueError(f"targets[{batch_index}]['image_id'] 不能为空")

        if image_id not in sample_cache:
            image_path, annotation_path = _resolve_sample_paths(image_id)
            with Image.open(image_path) as source_image:
                original_image = source_image.convert("RGB")
            sample_cache[image_id] = (original_image, _read_gt_boxes(annotation_path))

        original_image, original_gt_boxes = sample_cache[image_id]
        image, gt_boxes = _prepare_visualization_canvas(
            original_image,
            original_gt_boxes,
            validated_canvas_sizes[batch_index],
        )
        annotations: list[
            tuple[int, tuple[float, float, float, float], int]
        ] = []
        for weak_box_index in weak_box_indices:
            weak_box = tuple(float(value) for value in boxes_cpu[weak_box_index])
            class_id = int(class_ids[weak_box_index].item())
            class_cam = cams_by_box[weak_box_index, class_id]

            _overlay_cam(image, class_cam, weak_box)
            annotations.append((weak_box_index, weak_box, class_id))

        _draw_annotations(image, gt_boxes, annotations)

        safe_image_id = Path(image_id).name
        output_path = output_dir / f"{safe_image_id}.jpg"
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        image.save(temporary_path, format="JPEG", quality=95)
        os.replace(temporary_path, output_path)
        output_paths.append(output_path)

    return output_paths
