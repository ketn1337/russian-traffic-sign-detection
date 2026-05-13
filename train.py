# train.py
# pip install ultralytics kagglehub pillow tqdm pyyaml

import json
import shutil
from pathlib import Path

import kagglehub
import yaml
from tqdm import tqdm
from ultralytics import YOLO


DATASET_SLUG = "watchman/rtsd-dataset"

YOLO_DATASET_DIR = Path("data/rtsd-yolo")

USE_REDUCED_TRAIN = False  # True: train_anno_reduced.json, False: train_anno.json

MODEL_NAME = "yolov8n.pt"  # можно заменить на yolov8s.pt / yolov8m.pt
EPOCHS = 20
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4

PROJECT_DIR = "runs"
RUN_NAME = "rtsd_yolov8n_finetune"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_dataset_path() -> Path:
    dataset_path = Path.home() / ".cache/kagglehub/datasets/watchman/rtsd-dataset/versions/3"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset cache not found: {dataset_path}")

    print(f"Dataset path: {dataset_path}")
    return dataset_path


def load_coco(dataset_path: Path, anno_file: str) -> dict:
    anno_path = dataset_path / anno_file

    with open(anno_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    required_keys = ["images", "annotations", "categories"]
    missing = [key for key in required_keys if key not in coco]

    if missing:
        raise ValueError(f"{anno_file} is not valid COCO. Missing keys: {missing}")

    print(
        f"{anno_file}: "
        f"images={len(coco['images'])}, "
        f"annotations={len(coco['annotations'])}, "
        f"categories={len(coco['categories'])}"
    )

    return coco


def build_image_index(dataset_path: Path) -> dict:
    """
    Индексирует все изображения внутри rtsd-frames.
    Это нужно, потому что file_name в COCO может не совпадать 1-в-1
    с тем, как файл лежит в папке.
    """

    frames_dir = dataset_path / "rtsd-frames"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    image_index = {}

    for path in frames_dir.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS:
            # Ключ вида: rtsd-frames/xxx.jpg
            image_index[str(path.relative_to(dataset_path)).replace("\\", "/")] = path

            # Ключ вида: xxx.jpg или subdir/xxx.jpg относительно rtsd-frames
            image_index[str(path.relative_to(frames_dir)).replace("\\", "/")] = path

            # Ключ только по имени файла
            image_index[path.name] = path

    print(f"Indexed image keys: {len(image_index)}")

    if not image_index:
        raise RuntimeError(f"No images found in {frames_dir}")

    return image_index


def resolve_image_path(file_name: str, image_index: dict) -> Path | None:
    file_name = str(file_name).replace("\\", "/")
    name_only = Path(file_name).name

    candidates = [
        file_name,
        name_only,
        f"rtsd-frames/{name_only}",
    ]

    for candidate in candidates:
        if candidate in image_index:
            return image_index[candidate]

    return None


def build_category_mapping(train_coco: dict, val_coco: dict):
    """
    COCO category_id обычно идёт как 1..155.
    YOLO требует классы 0..N-1.
    """

    categories = {}

    for coco in [train_coco, val_coco]:
        for cat in coco["categories"]:
            categories[int(cat["id"])] = str(cat["name"])

    sorted_category_ids = sorted(categories.keys())

    category_id_to_yolo_id = {
        category_id: idx
        for idx, category_id in enumerate(sorted_category_ids)
    }

    class_names = [
        categories[category_id]
        for category_id in sorted_category_ids
    ]

    return category_id_to_yolo_id, class_names


def coco_bbox_to_yolo_bbox(bbox, image_width: float, image_height: float):
    """
    COCO bbox:
      [x_min, y_min, width, height]

    YOLO bbox:
      x_center y_center width height
      все значения нормализованы в диапазон [0, 1]
    """

    x, y, w, h = map(float, bbox)

    x_center = x + w / 2.0
    y_center = y + h / 2.0

    x_center /= image_width
    y_center /= image_height
    w /= image_width
    h /= image_height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)

    return x_center, y_center, w, h


def safe_symlink_or_copy(src: Path, dst: Path):
    """
    Сначала пытается сделать symlink, чтобы не копировать изображения.
    Если symlink невозможен, делает обычную копию.
    """

    if dst.exists() or dst.is_symlink():
        return

    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def prepare_split(
    dataset_path: Path,
    coco: dict,
    split_name: str,
    category_id_to_yolo_id: dict,
    image_index: dict,
):
    images_output_dir = YOLO_DATASET_DIR / "images" / split_name
    labels_output_dir = YOLO_DATASET_DIR / "labels" / split_name

    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {
        int(img["id"]): img
        for img in coco["images"]
    }

    annotations_by_image_id = {}

    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        annotations_by_image_id.setdefault(image_id, []).append(ann)

    converted_images = 0
    converted_annotations = 0
    skipped_annotations = 0
    missing_images = 0

    for image_id, image_info in tqdm(images_by_id.items(), desc=f"prepare {split_name}"):
        file_name = image_info["file_name"]

        src_image_path = resolve_image_path(file_name, image_index)

        if src_image_path is None:
            missing_images += 1

            if missing_images <= 10:
                print(f"Missing image: {file_name}")

            continue

        dst_image_path = images_output_dir / src_image_path.name
        safe_symlink_or_copy(src_image_path, dst_image_path)

        image_width = float(image_info["width"])
        image_height = float(image_info["height"])

        label_path = labels_output_dir / f"{dst_image_path.stem}.txt"

        yolo_lines = []

        for ann in annotations_by_image_id.get(image_id, []):
            category_id = int(ann["category_id"])

            if category_id not in category_id_to_yolo_id:
                skipped_annotations += 1
                continue

            bbox = ann.get("bbox")

            if bbox is None or len(bbox) != 4:
                skipped_annotations += 1
                continue

            x, y, w, h = map(float, bbox)

            if w <= 0 or h <= 0:
                skipped_annotations += 1
                continue

            yolo_class_id = category_id_to_yolo_id[category_id]

            x_center, y_center, yolo_w, yolo_h = coco_bbox_to_yolo_bbox(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            )

            if yolo_w <= 0 or yolo_h <= 0:
                skipped_annotations += 1
                continue

            yolo_lines.append(
                f"{yolo_class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{yolo_w:.6f} "
                f"{yolo_h:.6f}"
            )

            converted_annotations += 1

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        converted_images += 1

    print(f"\nSplit: {split_name}")
    print(f"Converted images: {converted_images}")
    print(f"Converted annotations: {converted_annotations}")
    print(f"Skipped annotations: {skipped_annotations}")
    print(f"Missing images: {missing_images}")

    if converted_images == 0:
        raise RuntimeError(
            f"No images were converted for split '{split_name}'. "
            f"Check file_name paths in annotations and rtsd-frames structure."
        )


def write_dataset_yaml(class_names):
    yaml_path = YOLO_DATASET_DIR / "data.yaml"

    data = {
        "path": str(YOLO_DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            idx: name
            for idx, name in enumerate(class_names)
        },
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"YOLO dataset config saved to: {yaml_path}")

    return yaml_path


def prepare_yolo_dataset(dataset_path: Path):
    train_anno_file = "train_anno_reduced.json" if USE_REDUCED_TRAIN else "train_anno.json"

    train_coco = load_coco(dataset_path, train_anno_file)
    val_coco = load_coco(dataset_path, "val_anno.json")

    category_id_to_yolo_id, class_names = build_category_mapping(
        train_coco=train_coco,
        val_coco=val_coco,
    )

    print(f"YOLO classes: {len(class_names)}")

    if YOLO_DATASET_DIR.exists():
        print(f"Removing old YOLO dataset: {YOLO_DATASET_DIR}")
        shutil.rmtree(YOLO_DATASET_DIR)

    image_index = build_image_index(dataset_path)

    prepare_split(
        dataset_path=dataset_path,
        coco=train_coco,
        split_name="train",
        category_id_to_yolo_id=category_id_to_yolo_id,
        image_index=image_index,
    )

    prepare_split(
        dataset_path=dataset_path,
        coco=val_coco,
        split_name="val",
        category_id_to_yolo_id=category_id_to_yolo_id,
        image_index=image_index,
    )

    yaml_path = write_dataset_yaml(class_names)

    mapping_path = YOLO_DATASET_DIR / "category_mapping.json"

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "category_id_to_yolo_id": {
                    str(k): v
                    for k, v in category_id_to_yolo_id.items()
                },
                "class_names": class_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Category mapping saved to: {mapping_path}")

    return yaml_path


def train_yolo(yaml_path: Path):
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        pretrained=True,
        device=device,
    )

    return results


def main():
    dataset_path = get_dataset_path()
    yaml_path = prepare_yolo_dataset(dataset_path)
    train_yolo(yaml_path)

    print("\nDone.")
    print("Best weights:")
    print(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print("Last weights:")
    print(f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt")


if __name__ == "__main__":
    main()
