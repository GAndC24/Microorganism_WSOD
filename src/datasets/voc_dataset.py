# Pascal VOC Style Dataset
import os
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import tv_tensors

from .utils.data_catalog import DataCatalog
from .utils.annotation_process import parse_voc_xml


Box = Tuple[float, float, float, float]   # (x1, y1, x2, y2)


def _detection_collate_fn(batch : List)-> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    '''batch : List of (image, target)'''
    images, targets = zip(*batch)
    return list(images), list(targets)


def _clip_and_filter_labeled_boxes(
    labeled_boxes: List[Dict[str, Any]],
    width: int,
    height: int,
) -> Tuple[List[Box], List[int]]:
    """裁剪标注框并过滤非有限值或零面积框。"""
    boxes: List[Box] = []
    labels: List[int] = []

    for item in labeled_boxes:
        x1, y1, x2, y2 = item["box"]
        coords = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        if not torch.isfinite(coords).all():
            continue

        x1 = float(max(0.0, min(float(width), float(x1))))
        y1 = float(max(0.0, min(float(height), float(y1))))
        x2 = float(max(0.0, min(float(width), float(x2))))
        y2 = float(max(0.0, min(float(height), float(y2))))

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        if x_max <= x_min or y_max <= y_min:
            continue

        boxes.append((x_min, y_min, x_max, y_max))
        labels.append(int(item["class_id"]))

    return boxes, labels


class VocDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,      # dataset name
        split: str,       # dataset split, "train" or "val" or "test"
        target_mode: str,    # target mode, "gt" or "wb"
        transforms: Optional[Callable] = None,      # data transforms
        use_bg_boxes: bool = False,   # 是否加载背景框
    )-> None:
        super().__init__()

        data_catalog = DataCatalog()
        self.dataset_info = data_catalog.get(dataset_name)

        self.transforms = transforms

        # get annotation directory
        if target_mode == "wb":
            ann_path = self.dataset_info["ann_dir_wb"]
        else:
            ann_path = self.dataset_info["ann_dir_gt"]
        self.ann_dir = ann_path

        # get ImageSet directory
        if split == "train":
            set_dir_path = self.dataset_info["train_split"]
        elif split == "val":
            set_dir_path = self.dataset_info["val_split"]
        else:
            set_dir_path = self.dataset_info["test_split"]
        self.set_dir = set_dir_path  # dataset split folder

        # get image directory
        self.img_dir = self.dataset_info["img_dir"]  # jpg images folder

        # load images
        imageset_txt = self.set_dir
        if not os.path.isfile(imageset_txt):
            raise FileNotFoundError(f"ImageSet file not found: {imageset_txt}")
        with open(imageset_txt, "r", encoding="utf-8") as f:
            self.img_ids = [line.strip() for line in f.readlines() if line.strip()]

        self.bg_boxes = None
        if use_bg_boxes:
            if "bg_boxes" not in self.dataset_info:
                raise KeyError(f"数据集 {dataset_name} 未配置 bg_boxes。")
            self.bg_boxes = torch.load(
                self.dataset_info["bg_boxes"],
                map_location="cpu",
                weights_only=True,
            )


    def __len__(self) -> int:
        return len(self.img_ids)


    def __getitem__(self, idx: int)-> Tuple[torch.Tensor, Dict[str, Any]]:
        image_id = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        xml_path = os.path.join(self.ann_dir, f"{image_id}.xml")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        image_pil = Image.open(img_path).convert("RGB")
        W, H = image_pil.size  # PIL: (W, H)

        image_info, ann_boxes = parse_voc_xml(xml_path, self.dataset_info["class_map_encoding"])

        ann_boxes, labels = _clip_and_filter_labeled_boxes(ann_boxes, W, H)

        ann_boxes_tensor = torch.tensor(ann_boxes, dtype=torch.float32)  # [N,4]
        labels_tensor = torch.tensor(labels, dtype=torch.int64)  # [N]
        image = tv_tensors.Image(image_pil)
        ann_boxes_tv = tv_tensors.BoundingBoxes(
            ann_boxes_tensor,
            format="XYXY",
            canvas_size=(H, W)
        )
        target: Dict[str, Any] = {
            "boxes": ann_boxes_tv,
            "labels": labels_tensor,
            "image_id": image_id
        }

        if self.bg_boxes is not None:
            records = self.bg_boxes.get("records", {})
            if image_id not in records:
                raise KeyError(f"背景框文件中缺少图像 {image_id} 的记录。")
            bg_boxes_tensor = records[image_id]["boxes"].reshape(-1, 4).to(dtype=torch.float32)
            target["bg_boxes"] = tv_tensors.BoundingBoxes(
                bg_boxes_tensor,
                format="XYXY",
                canvas_size=(H, W),
            )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def build_voc_dataloader(
    dataset_name: str,      # dataset name
    split: str,       # dataset split, "train" or "val" or "test"
    target_mode: str,    # target mode, "gt" or "wb"
    batch_size: int,
    transforms: Optional[Callable] = None,      # data transforms
    use_bg_boxes: bool = False,   # 是否加载背景框
)->DataLoader:
    dataset = VocDataset(
        dataset_name,
        split,
        target_mode,
        transforms=transforms,
        use_bg_boxes=use_bg_boxes,
    )

    if split == "train":
        shuffle = True
    else:
        shuffle = False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True, collate_fn=_detection_collate_fn)

    return dataloader
