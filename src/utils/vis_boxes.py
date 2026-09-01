"""伪标签与真实标注框可视化。"""

from __future__ import annotations

import math
import os
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Collection, Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from ..datasets.utils.data_catalog import DataCatalog


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_PSEUDO_COLOR = (255, 45, 45)
_HIGHLIGHT_COLOR = (255, 215, 0)
_GT_COLOR = (40, 210, 80)


def _resolve_sample_paths(img_id: str) -> tuple[Path, Path]:
    """从数据目录表中定位图片及其 GT XML。"""
    catalog = DataCatalog()
    candidates: list[tuple[Path, Path]] = []

    for dataset_name, config in catalog.catalog.get("DATASETS", {}).items():
        if "img_dir" not in config or "ann_dir_gt" not in config:
            continue

        dataset_info = catalog.get(dataset_name)
        image_dir = Path(dataset_info["img_dir"])
        annotation_path = Path(dataset_info["ann_dir_gt"]) / f"{img_id}.xml"
        image_paths = [image_dir / f"{img_id}{suffix}" for suffix in _IMAGE_SUFFIXES]
        candidates.extend((image_path, annotation_path) for image_path in image_paths)

        for image_path in image_paths:
            if image_path.is_file() and annotation_path.is_file():
                return image_path, annotation_path

    searched = "\n".join(
        f"  image={image_path}, annotation={annotation_path}"
        for image_path, annotation_path in candidates
    )
    raise FileNotFoundError(
        f"无法为图片 {img_id!r} 找到原图及 GT 标注。已检查：\n{searched}"
    )


def _to_cpu_list(value: object, name: str) -> list[object]:
    """将 Tensor 或序列转换为 CPU 列表。"""
    converted = value
    for method_name in ("detach", "cpu", "tolist"):
        method = getattr(converted, method_name, None)
        if method is not None:
            converted = method()

    if converted is None:
        return []
    if isinstance(converted, (str, bytes)) or not isinstance(converted, Sequence):
        raise TypeError(f"{name} 必须是 Tensor 或序列")

    return list(converted)


def _to_pseudo_label_list(
    boxes: object,
    scores: object,
    labels: object,
) -> list[
    tuple[int, tuple[float, float, float, float], float, int]
]:
    """转换并对齐伪标签的原始索引、坐标、分数和类别 ID。"""
    box_rows = _to_cpu_list(boxes, "boxes")
    score_values = _to_cpu_list(scores, "scores")
    label_values = _to_cpu_list(labels, "labels")

    is_single_box = len(box_rows) == 4 and all(
        isinstance(item, (int, float)) for item in box_rows
    )
    if is_single_box:
        box_rows = [box_rows]

    if not (len(box_rows) == len(score_values) == len(label_values)):
        raise ValueError(
            "boxes、scores 和 labels 的数量必须一致，"
            f"当前分别为 {len(box_rows)}、{len(score_values)}、{len(label_values)}"
        )

    result: list[
        tuple[int, tuple[float, float, float, float], float, int]
    ] = []
    for box_index, (row, raw_score, raw_label) in enumerate(
        zip(box_rows, score_values, label_values)
    ):
        if not isinstance(row, Sequence) or len(row) != 4:
            raise ValueError(f"boxes[{box_index}] 必须恰好包含 4 个 XYXY 坐标")

        x1, y1, x2, y2 = (float(coord) for coord in row)
        coords = (x1, y1, x2, y2)
        if not all(math.isfinite(coord) for coord in coords):
            warnings.warn(f"跳过包含非有限坐标的伪标签框：{coords}", RuntimeWarning)
            continue

        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError(f"scores[{box_index}] 必须是有限数值，当前为 {score}")

        label_value = float(raw_label)
        if not math.isfinite(label_value) or not label_value.is_integer():
            raise ValueError(
                f"labels[{box_index}] 必须是整数类别 ID，当前为 {raw_label!r}"
            )

        result.append((box_index, coords, score, int(label_value)))

    return result


def select_batch_highest_score_indices(
    batch_boxes: Sequence[object],
    batch_scores: Sequence[object],
    batch_labels: Sequence[object],
) -> list[frozenset[int]]:
    """
    找出整个 batch 中每个类别分数最高的伪标签框。

    返回列表与 batch 图片顺序一致，每个元素保存对应图片中需要显示
    类别 ID 和分数的原始框索引。最高分相同时保留最先出现的框。
    """
    batch_size = len(batch_boxes)
    if not (batch_size == len(batch_scores) == len(batch_labels)):
        raise ValueError(
            "batch_boxes、batch_scores 和 batch_labels 包含的图片数量必须一致"
        )

    highest_by_class: dict[int, tuple[float, int, int]] = {}
    for image_index, (boxes, scores, labels) in enumerate(
        zip(batch_boxes, batch_scores, batch_labels)
    ):
        pseudo_labels = _to_pseudo_label_list(boxes, scores, labels)
        for box_index, _, score, class_id in pseudo_labels:
            current = highest_by_class.get(class_id)
            if current is None or score > current[0]:
                highest_by_class[class_id] = (score, image_index, box_index)

    highlighted_indices = [set() for _ in range(batch_size)]
    for _, image_index, box_index in highest_by_class.values():
        highlighted_indices[image_index].add(box_index)

    return [frozenset(indices) for indices in highlighted_indices]


def _read_gt_boxes(
    annotation_path: Path,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """读取 Pascal VOC XML 中的类别名和 XYXY 框。"""
    root = ET.parse(annotation_path).getroot()
    boxes: list[tuple[str, tuple[float, float, float, float]]] = []

    for index, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "unknown").strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            raise ValueError(
                f"GT 标注 {annotation_path} 的第 {index + 1} 个 object 缺少 bndbox"
            )

        try:
            coords = tuple(
                float(bndbox.findtext(key))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"GT 标注 {annotation_path} 的第 {index + 1} 个框坐标无效"
            ) from exc

        boxes.append((name, coords))

    return boxes


def _clip_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """裁剪并规范化框坐标，过滤零面积框。"""
    x1, y1, x2, y2 = box
    left = max(0.0, min(float(width - 1), min(x1, x2)))
    top = max(0.0, min(float(height - 1), min(y1, y2)))
    right = max(0.0, min(float(width - 1), max(x1, x2)))
    bottom = max(0.0, min(float(height - 1), max(y1, y2)))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    line_width: int,
    label: str | None = None,
) -> None:
    draw.rectangle(box, outline=color, width=line_width)
    if label is None:
        return

    text_box = draw.textbbox((box[0], box[1]), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_top = max(0.0, box[1] - text_height - 4)
    draw.rectangle(
        (box[0], label_top, box[0] + text_width + 4, label_top + text_height + 4),
        fill=color,
    )
    draw.text((box[0] + 2, label_top + 2), label, fill=(255, 255, 255))


def visualize(
    img_id: str,
    boxes: object,
    scores: object,
    labels: object,
    log_dir: str | os.PathLike[str],
    epoch: int,
    batch_iter: int,
    highlighted_box_indices: Collection[int],
    box_canvas_size: tuple[int, int] | None = None,
) -> Path:
    """
    在原图上绘制伪标签框和 Pascal VOC GT 框并保存。

    红色框表示普通伪标签，黄色框表示 batch 内同类别分数最高的伪标签，
    绿色框表示 GT。最高分伪标签框同时显示类别 ID 和分数。
    ``box_canvas_size`` 为伪标签坐标系的 ``(高度, 宽度)``；未传入时，
    默认 ``boxes`` 已经使用原图坐标系。
    """
    image_id = str(img_id).strip()
    if not image_id:
        raise ValueError("img_id 不能为空")

    image_path, annotation_path = _resolve_sample_paths(image_id)
    pseudo_labels = _to_pseudo_label_list(boxes, scores, labels)
    gt_boxes = _read_gt_boxes(annotation_path)
    highlighted_indices = set(highlighted_box_indices)

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    image_width, image_height = image.size
    if box_canvas_size is not None:
        canvas_height, canvas_width = box_canvas_size
        if canvas_height <= 0 or canvas_width <= 0:
            raise ValueError("box_canvas_size 的高度和宽度必须为正数")
        scale_x = image_width / canvas_width
        scale_y = image_height / canvas_height
        pseudo_labels = [
            (
                box_index,
                (
                    box[0] * scale_x,
                    box[1] * scale_y,
                    box[2] * scale_x,
                    box[3] * scale_y,
                ),
                score,
                class_id,
            )
            for box_index, box, score, class_id in pseudo_labels
        ]

    available_indices = {box_index for box_index, _, _, _ in pseudo_labels}
    unknown_indices = highlighted_indices - available_indices
    if unknown_indices:
        raise ValueError(f"需要高亮的框索引不存在：{sorted(unknown_indices)}")

    line_width = max(2, round(min(image_width, image_height) / 250))
    draw = ImageDraw.Draw(image)

    for class_name, gt_box in gt_boxes:
        clipped_box = _clip_box(gt_box, image_width, image_height)
        if clipped_box is not None:
            _draw_labeled_box(
                draw,
                clipped_box,
                _GT_COLOR,
                line_width,
                label=f"GT: {class_name}",
            )

    for box_index, pseudo_box, score, class_id in pseudo_labels:
        clipped_box = _clip_box(pseudo_box, image_width, image_height)
        if clipped_box is not None:
            is_highlighted = box_index in highlighted_indices
            pseudo_color = _HIGHLIGHT_COLOR if is_highlighted else _PSEUDO_COLOR
            pseudo_text = (
                f"label: {class_id}, score: {score:.4f}"
                if is_highlighted
                else None
            )
            _draw_labeled_box(
                draw,
                clipped_box,
                pseudo_color,
                line_width,
                label=pseudo_text,
            )

    legend_text = "Pseudo label    Top score    GT"
    legend_box = draw.textbbox((6, 6), legend_text)
    draw.rectangle((3, 3, legend_box[2] + 9, legend_box[3] + 9), fill=(0, 0, 0))
    draw.text((6, 6), "Pseudo label", fill=_PSEUDO_COLOR)
    top_score_x = 6 + draw.textlength("Pseudo label    ")
    draw.text((top_score_x, 6), "Top score", fill=_HIGHLIGHT_COLOR)
    gt_x = 6 + draw.textlength("Pseudo label    Top score    ")
    draw.text((gt_x, 6), "GT", fill=_GT_COLOR)

    output_dir = (
        Path(log_dir)
        / "visualize_boxes"
        / f"epoch_{epoch:03d}"
        / f"batch_{batch_iter:05d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(image_id).name}.jpg"
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    image.save(temporary_path, format="JPEG", quality=95)
    os.replace(temporary_path, output_path)

    return output_path
