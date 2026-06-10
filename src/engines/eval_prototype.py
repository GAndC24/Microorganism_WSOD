import argparse
import os
import torch
import yaml
from typing import Dict, Any, List
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import numpy as np
from datetime import datetime

from ..datasets.voc_dataset import build_voc_dataloader
from ..models.prototype_evaluator import build_prototype_evaluator


def _load_yaml(config_path : str)-> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def _build_boxes(targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    """
    Construct boxes tensor from targets.
    :param targets: List of dictionaries containing bounding box information.
    :return: GT boxes tensor of shape [R, 5], where each box is [batch_idx, x1, y1, x2, y2].
    """
    gt_boxes = []
    for batch_idx, target in enumerate(targets):
        boxes = target["boxes"]  # Convert to [x1, y1, x2, y2]
        batch_indices = torch.full((boxes.size(0), 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
        gt_boxes.append(torch.cat([batch_indices, boxes], dim=1))  # Combine batch_idx with boxes

    return torch.cat(gt_boxes, dim=0)  # Concatenate all boxes across the batch


def _build_boxes_label(targets: List[Dict[str, torch.Tensor]], num_classes: int) -> torch.Tensor:
    """
    construct one-hot labels for a batch of boxes
    :param targets: image annotations
    :param num_classes : number of classes
    :return: labels : one-hot tensor, [R = num_boxes, num_classes]
    """
    all_labels = []
    for target in targets:
        all_labels.append(target["labels"])
    all_labels = torch.cat(all_labels, dim=0)  # [R]
    one_hot_labels = torch.zeros((all_labels.size(0), num_classes), dtype=torch.float32).to(all_labels.device)
    one_hot_labels.scatter_(1, all_labels.unsqueeze(1), 1.0)
    return one_hot_labels


def _evaluate_similarity_and_margin(model, loader, prototypes, num_classes, device, save_dir, split_name):
    os.makedirs(save_dir, exist_ok=True)

    # 全局加权统计：sum/cnt
    global_sum = {k: 0.0 for k in range(num_classes)}
    global_cnt = {k: 0   for k in range(num_classes)}

    # 分布统计（用于画图与分位数）
    pos_list = defaultdict(list)
    negmax_list = defaultdict(list)
    margin_list = defaultdict(list)

    model.eval()
    num_iters = len(loader)
    pbar = tqdm(
        enumerate(loader, start=1),
        total=num_iters,
        desc=f"Computing",
        leave=False,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for iter, (images, target) in pbar:
            images = [img.to(device) for img in images]
            x = torch.stack(images, dim=0)
            targets = [{k: v.to(device) for k, v in t.items()} for t in target]

            boxes = _build_boxes(targets)
            boxes_labels = _build_boxes_label(targets, num_classes)

            out = model(x, boxes, boxes_labels, prototypes, return_details=True)

            # 1) 全局加权
            for k in range(num_classes):
                global_sum[k] += float(out["sum"][k])
                global_cnt[k] += int(out["cnt"][k])

            # 2) 分布数据
            details = out["details"]
            for k in range(num_classes):
                pos_list[k].extend(details["pos"][k])
                negmax_list[k].extend(details["neg_max"][k])
                margin_list[k].extend(details["margin"][k])

    # 计算全局加权均值
    mean_pos = {}
    for k in range(num_classes):
        mean_pos[k] = (global_sum[k] / global_cnt[k]) if global_cnt[k] > 0 else 0.0

    # 输出与保存统计表（csv）
    csv_path = os.path.join(save_dir, f"{split_name}_proto_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_id", "cnt",
            "mean_pos",
            "mean_margin", "p25_margin", "p50_margin", "p75_margin",
            "pos_rate(sim>0)", "margin_rate(margin>0)"
        ])
        for k in range(num_classes):
            cnt = global_cnt[k]
            if cnt == 0:
                writer.writerow([k, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                continue

            m = np.array(margin_list[k], dtype=np.float32)
            p = np.array(pos_list[k], dtype=np.float32)

            writer.writerow([
                k, cnt,
                float(mean_pos[k]),
                float(m.mean()), float(np.percentile(m, 25)), float(np.percentile(m, 50)), float(np.percentile(m, 75)),
                float((p > 0).mean()), float((m > 0).mean())
            ])

    # 画分布图：pos 与 margin（每类两张图）
    for k in range(num_classes):
        if len(pos_list[k]) == 0:
            continue

        # pos hist
        plt.figure()
        plt.hist(pos_list[k], bins=40)
        plt.title(f"{split_name} Class {k} - sim_pos distribution (cnt={len(pos_list[k])})")
        plt.xlabel("cosine(sim_pos)")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_class{k}_pos_hist.png"))
        plt.close()

        # margin hist
        plt.figure()
        plt.hist(margin_list[k], bins=40)
        plt.title(f"{split_name} Class {k} - margin=pos-maxneg (cnt={len(margin_list[k])})")
        plt.xlabel("margin")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{split_name}_class{k}_margin_hist.png"))
        plt.close()

    return mean_pos, global_cnt, csv_path, save_dir


def eval(
    config_path : str,
)-> None:
    cfg = _load_yaml(config_path)

    # -----Init Dataloader-----
    # data preprocessing transforms
    img_size = cfg['DATA']["IMG_SIZE"]
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = build_voc_dataloader(
        dataset_name=cfg['DATA']['DATASET_NAME'],
        split='train',
        target_mode='gt',
        batch_size=cfg['DATA']['BATCH_SIZE'],
        transforms=transform,
    )
    val_loader = build_voc_dataloader(
        dataset_name=cfg['DATA']['DATASET_NAME'],
        split='val',
        target_mode='gt',
        batch_size=cfg['DATA']['BATCH_SIZE'],
        transforms=transform,
    )
    # test_loader = build_voc_dataloader(
    #     dataset_name=cfg['DATA']['DATASET_NAME'],
    #     split='test',
    #     target_mode='gt',
    #     batch_size=cfg['DATA']['BATCH_SIZE'],
    #     transforms=transform,
    # )


    # -----Init Model-----
    model = build_prototype_evaluator(cfg)
    model = model.to(cfg['RUNTIME']['DEVICE'])
    model.eval()

    # -----Load Dataset Morphological Prototypes-----
    prototypes = torch.load(cfg['MODEL']['DATASET_MPS_PATH'])
    prototypes = {k: v.to(cfg['RUNTIME']['DEVICE']) for k, v in prototypes.items()}


    # -----Evaluate similarity and margin distributions-----
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"./results/proto_eval/{start_time}"
    train_mean_pos, train_cnt, train_csv, train_dir = _evaluate_similarity_and_margin(
        model, train_loader, prototypes, cfg['DATA']['NUM_CLASSES'], cfg['RUNTIME']['DEVICE'], save_dir=save_dir, split_name="train"
    )
    val_mean_pos, val_cnt, val_csv, val_dir = _evaluate_similarity_and_margin(
        model, val_loader, prototypes, cfg['DATA']['NUM_CLASSES'], cfg['RUNTIME']['DEVICE'], save_dir=save_dir, split_name="val"
    )
    # test_mean_pos, test_cnt, test_csv, test_dir = _evaluate_similarity_and_margin(
    #     model, test_loader, prototypes, cfg['DATA']['NUM_CLASSES'], cfg['RUNTIME']['DEVICE'], save_dir=save_dir, split_name="test"
    # )
    print("\n-----Prototype Similarity & Margin Evaluation Results-----")
    print(train_mean_pos, train_cnt, train_csv, train_dir)
    print(val_mean_pos, val_cnt, val_csv, val_dir)
    # print(test_mean_pos, test_cnt, test_csv, test_dir)