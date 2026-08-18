'''
evaluate engine for Prototype
run: 
    python -m src.engines.eval_prototype --config "src/configs/cfg_prototype_builder.yaml"
'''
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
import numpy as np
import torch
import os
import yaml
from torchvision.transforms import v2 as T
from tqdm.auto import tqdm
import argparse

from ..datasets.voc_dataset import build_voc_dataloader
from ..models.prototype_builder import build_prototype_builder_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training Config")

    p.add_argument('--config', type=str, required=True, help='config file path')

    return p.parse_args()


def _load_yaml(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("配置请求使用 cuda，但当前环境不可用，已切换到 cpu。")
        return torch.device("cpu")
    return torch.device(device_name)


def _build_image_size(cfg: Dict[str, Any]) -> Tuple[int, int]:
    data_cfg = cfg["DATA"]
    if "IMG_SIZE" in data_cfg:
        img_size = int(data_cfg["IMG_SIZE"])
        return img_size, img_size
    return int(data_cfg["IMG_H"]), int(data_cfg["IMG_W"])


def _load_checkpoint(path: Optional[Path], device: torch.device) -> Optional[Dict[str, Any]]:
    if path is None:
        print("未找到 checkpoint，模型将使用初始化权重进行评估。")
        return None
    checkpoint = torch.load(path, map_location=device)
    print(f"Loaded checkpoint: {path}")
    return checkpoint


def _load_model_weights(model: torch.nn.Module, checkpoint: Optional[Dict[str, Any]]) -> None:
    if checkpoint is None:
        return
    if "model_state_dict" not in checkpoint:
        raise KeyError("checkpoint 中缺少 model_state_dict。")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def _load_prototypes(
    checkpoint: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    if checkpoint is not None and "dataset_MPs" in checkpoint:
        prototypes = checkpoint["dataset_MPs"]
        print("Loaded dataset MPs from checkpoint.")
    else:
        raise FileNotFoundError("未找到 dataset_MPs。")

    return {int(k): v.to(device) for k, v in prototypes.items()}


def _build_boxes(
    targets: List[Dict[str, torch.Tensor]],
    box_key: str = "boxes",
) -> torch.Tensor:
    """
    从 targets 构造 RoIAlign 需要的框张量。
    :return: [R, 5]，每行格式为 [batch_idx, x1, y1, x2, y2]
    """
    roi_boxes = []
    for batch_idx, target in enumerate(targets):
        boxes = target[box_key]
        if boxes.numel() == 0:
            continue
        batch_indices = torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
        roi_boxes.append(torch.cat([batch_indices, boxes], dim=1))

    if not roi_boxes:
        device = targets[0][box_key].device if targets else torch.device("cpu")
        return torch.empty((0, 5), dtype=torch.float32, device=device)
    return torch.cat(roi_boxes, dim=0)


def _build_boxes_label(targets: List[Dict[str, torch.Tensor]], num_classes_with_bg: int) -> torch.Tensor:
    """
    为 GT boxes 构造 one-hot 标签。0 预留给背景，前景类别整体右移一位。
    :return: [R, num_classes_with_bg]
    """
    all_labels = []
    for target in targets:
        labels = target["labels"]
        if labels.numel() == 0:
            continue
        all_labels.append(labels + 1)

    if not all_labels:
        device = targets[0]["labels"].device if targets else torch.device("cpu")
        return torch.empty((0, num_classes_with_bg), dtype=torch.float32, device=device)

    all_labels = torch.cat(all_labels, dim=0)
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes_with_bg), dtype=torch.float32, device=all_labels.device)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels


def _build_bg_boxes_label(
    num_boxes: int,
    num_classes_with_bg: int,
    device: torch.device,
) -> torch.Tensor:
    """为背景框构造类别 0 的 one-hot 标签。"""
    labels = torch.zeros(
        (num_boxes, num_classes_with_bg),
        dtype=torch.float32,
        device=device,
    )
    if num_boxes > 0:
        labels[:, 0] = 1.0
    return labels


def _safe_stats(values: List[float]) -> Dict[str, float]:
    if len(values) == 0:
        return {"mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "gt0_rate": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "gt0_rate": float((arr > 0).mean()),
    }


def _plot_hist(values: List[float], title: str, xlabel: str, save_path: Path) -> None:
    if len(values) == 0:
        return
    plt.figure()
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _evaluate_gt_similarity_and_margin(
    model: torch.nn.Module,
    loader,
    prototypes: Dict[int, torch.Tensor],
    num_classes_with_bg: int,
    device: torch.device,
    save_dir: Path,
    split_name: str,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    global_sum = {k: 0.0 for k in range(num_classes_with_bg)}
    global_cnt = {k: 0 for k in range(num_classes_with_bg)}

    pos_list = defaultdict(list)
    margin_list = defaultdict(list)
    bg_list = defaultdict(list)

    model.eval()
    pbar = tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        desc=f"Computing {split_name}",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for _, (images, target) in pbar:
            images = [img.to(device) for img in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in target
            ]

            boxes = _build_boxes(targets)
            boxes_labels = _build_boxes_label(targets, num_classes_with_bg)
            if boxes.size(0) == 0:
                continue

            x = torch.stack(images, dim=0)
            out = model.eval_prototype(x, boxes, boxes_labels, prototypes, return_details=True)

            for k in range(num_classes_with_bg):
                global_sum[k] += float(out["sum"][k])
                global_cnt[k] += int(out["cnt"][k])

            details = out["details"]
            for k in range(num_classes_with_bg):
                pos_list[k].extend(details["pos"][k])
                margin_list[k].extend(details["margin"][k])
                bg_list[k].extend(details["bg"][k])

    mean_pos = {
        k: (global_sum[k] / global_cnt[k]) if global_cnt[k] > 0 else 0.0
        for k in range(num_classes_with_bg)
    }

    csv_path = save_dir / f"{split_name}_proto_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "cnt",
            "mean_pos",
            "mean_bg", "p25_bg", "p50_bg", "p75_bg",
            "mean_margin", "p25_margin", "p50_margin", "p75_margin",
            "pos_rate(sim>0)", "bg_rate(sim>0)", "margin_rate(margin>0)",
        ])
        for k in range(num_classes_with_bg):
            cnt = global_cnt[k]
            if cnt == 0:
                writer.writerow([k, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            pos_stats = _safe_stats(pos_list[k])
            bg_stats = _safe_stats(bg_list[k])
            margin_stats = _safe_stats(margin_list[k])
            writer.writerow([
                k, cnt,
                float(mean_pos[k]),
                bg_stats["mean"], bg_stats["p25"], bg_stats["p50"], bg_stats["p75"],
                margin_stats["mean"], margin_stats["p25"], margin_stats["p50"], margin_stats["p75"],
                pos_stats["gt0_rate"], bg_stats["gt0_rate"], margin_stats["gt0_rate"],
            ])

    for k in range(num_classes_with_bg):
        _plot_hist(
            pos_list[k],
            f"{split_name} Class {k} - sim_pos distribution (cnt={len(pos_list[k])})",
            "cosine(sim_pos)",
            save_dir / f"{split_name}_class{k}_pos_hist.png",
        )
        _plot_hist(
            bg_list[k],
            f"{split_name} Class {k} - sim_bg distribution (cnt={len(bg_list[k])})",
            "cosine(sim_bg)",
            save_dir / f"{split_name}_class{k}_bg_hist.png",
        )
        _plot_hist(
            margin_list[k],
            f"{split_name} Class {k} - margin=pos-maxneg (cnt={len(margin_list[k])})",
            "margin",
            save_dir / f"{split_name}_class{k}_margin_hist.png",
        )

    return mean_pos, global_cnt, csv_path, save_dir


def _evaluate_bg_positive_similarity(
    model: torch.nn.Module,
    loader,
    prototypes: Dict[int, torch.Tensor],
    num_classes_with_bg: int,
    device: torch.device,
    save_dir: Path,
    split_name: str,
):
    """仅评估背景框与所有类别原型之间的 positive similarity。"""
    save_dir.mkdir(parents=True, exist_ok=True)

    similarity_sum = {k: 0.0 for k in range(num_classes_with_bg)}
    box_count = {k: 0 for k in range(num_classes_with_bg)}
    positive_similarities = defaultdict(list)

    model.eval()
    pbar = tqdm(
        enumerate(loader, start=1),
        total=len(loader),
        desc=f"Computing {split_name} background boxes",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for _, (images, target) in pbar:
            images = [img.to(device) for img in images]
            targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in target
            ]

            boxes = _build_boxes(targets, box_key="bg_boxes")
            if boxes.size(0) == 0:
                continue
            boxes_labels = _build_bg_boxes_label(
                num_boxes=boxes.size(0),
                num_classes_with_bg=num_classes_with_bg,
                device=boxes.device,
            )

            x = torch.stack(images, dim=0)
            out = model.eval_prototype(
                x,
                boxes,
                boxes_labels,
                prototypes,
                return_details=True,
                positive_only=True,
            )
            for k in range(num_classes_with_bg):
                similarity_sum[k] += float(out["sum"][k])
                box_count[k] += int(out["cnt"][k])
                positive_similarities[k].extend(out["details"]["pos"][k])

    mean_positive_similarity = {
        k: similarity_sum[k] / box_count[k] if box_count[k] > 0 else 0.0
        for k in range(num_classes_with_bg)
    }
    csv_path = save_dir / f"{split_name}_bg_proto_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "cnt", "mean_pos"])
        for k in range(num_classes_with_bg):
            writer.writerow([k, box_count[k], mean_positive_similarity[k]])

    for k in range(num_classes_with_bg):
        _plot_hist(
            positive_similarities[k],
            (
                f"{split_name} Background Boxes vs Class {k} Prototype "
                f"- sim_pos distribution (cnt={box_count[k]})"
            ),
            "cosine(sim_pos)",
            save_dir / f"{split_name}_bg_class{k}_pos_hist.png",
        )

    return mean_positive_similarity, box_count, csv_path, save_dir


def eval(config_path: str) -> None:
    cfg = _load_yaml(config_path)
    device = _resolve_device(cfg["RUNTIME"]["DEVICE"])

    img_size = _build_image_size(cfg)
    transform = T.Compose([
        T.Resize(img_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    splits = ["train", "val"]
    gt_loaders = {
        split: build_voc_dataloader(
            dataset_name=cfg["DATA"]["DATASET_NAME"],
            split=split,
            target_mode="gt",
            batch_size=cfg["DATA"]["BATCH_SIZE"],
            transforms=transform,
        )
        for split in splits
    }
    bg_batch_size = int(cfg["EVAL"].get("BG_BATCH_SIZE", cfg["DATA"]["BATCH_SIZE"]))
    if bg_batch_size <= 0:
        raise ValueError("EVAL.BG_BATCH_SIZE 必须为正整数。")
    bg_loaders = {
        split: build_voc_dataloader(
            dataset_name=cfg["DATA"]["DATASET_NAME"],
            split=split,
            target_mode="gt",
            batch_size=bg_batch_size,
            transforms=transform,
            use_bg_boxes=True,
        )
        for split in splits
    }

    model = build_prototype_builder_model(cfg).to(device)
    checkpoint_path = cfg['EVAL']['LOAD_CHECKPOINT_PATH']
    checkpoint = _load_checkpoint(checkpoint_path, device)
    _load_model_weights(model, checkpoint)
    prototypes = _load_prototypes(checkpoint, device)

    expected_proto_keys = set(range(cfg["DATA"]["NUM_CLASSES"] + 1))
    if set(prototypes.keys()) != expected_proto_keys:
        raise ValueError(f"原型 key 应为 {sorted(expected_proto_keys)}，实际为 {sorted(prototypes.keys())}。")

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = Path("results") / "proto_eval" / start_time
    num_classes_with_bg = cfg["DATA"]["NUM_CLASSES"] + 1

    print("\n-----GT Box Prototype Similarity & Margin Evaluation Results-----")
    for split, loader in gt_loaders.items():
        mean_pos, cnt, csv_path, out_dir = _evaluate_gt_similarity_and_margin(
            model=model,
            loader=loader,
            prototypes=prototypes,
            num_classes_with_bg=num_classes_with_bg,
            device=device,
            save_dir=save_dir / "gt",
            split_name=split,
        )
        print(f"{split} GT: mean_pos={mean_pos}, cnt={cnt}, csv={csv_path}, dir={out_dir}")

    print("\n-----Background Box Positive Similarity Evaluation Results-----")
    for split, loader in bg_loaders.items():
        mean_pos, cnt, csv_path, out_dir = _evaluate_bg_positive_similarity(
            model=model,
            loader=loader,
            prototypes=prototypes,
            num_classes_with_bg=num_classes_with_bg,
            device=device,
            save_dir=save_dir / "bg",
            split_name=split,
        )
        print(
            f"{split} BG: mean_pos={mean_pos}, cnt={cnt}, "
            f"csv={csv_path}, dir={out_dir}"
        )


def main():
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    args = _get_args()

    eval(args.config)


if __name__ == '__main__':
    main()
