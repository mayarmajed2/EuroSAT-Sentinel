"""Dataset downloading, stratified splitting, transforms, and DataLoader construction for EuroSAT."""

from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.utils import ensure_dirs, setup_logger

logger = setup_logger("eurosat.data")


class EuroSATSubsetDataset(Dataset):
    """Custom PyTorch Dataset wrapping file paths and integer class labels with split-specific transforms."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        img_path = Path(record["filepath"])
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor_img = self.transform(rgb_img)
        else:
            tensor_img = transforms.ToTensor()(rgb_img)
        return tensor_img, int(record["label"])


def get_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """Create domain-specific training and evaluation transforms for Sentinel-2 overhead imagery.

    Overhead satellite imagery is rotationally invariant (there is no canonical 'up'),
    making horizontal/vertical flips and 90-degree rotations physically realistic augmentations.
    """
    input_size = int(config["data"]["model_input_size"])
    aug_cfg = config["augmentation"]
    mean = aug_cfg["imagenet_norm"]["mean"]
    std = aug_cfg["imagenet_norm"]["std"]

    train_transform_list = [
        transforms.Resize((input_size, input_size)),
    ]
    if aug_cfg.get("random_horizontal_flip", True):
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug_cfg.get("random_vertical_flip", True):
        train_transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    rot_deg = aug_cfg.get("random_rotation_degrees", 0)
    if rot_deg > 0:
        train_transform_list.append(transforms.RandomRotation(degrees=rot_deg))
    if "color_jitter" in aug_cfg:
        cj = aug_cfg["color_jitter"]
        train_transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0.1),
                contrast=cj.get("contrast", 0.1),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.02),
            )
        )
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    raw_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    return {
        "train": transforms.Compose(train_transform_list),
        "val": eval_transform,
        "test": eval_transform,
        "raw": raw_transform,
    }


def download_or_verify_eurosat(root_dir: str | Path, download: bool = True) -> Tuple[Path, List[str]]:
    """Download or verify the official EuroSAT RGB dataset via torchvision.datasets.EuroSAT.

    Falls back to a deterministic synthetic satellite benchmark dataset ONLY if offline/testing
    and raw files are absent.
    """
    root_path = Path(root_dir)
    ensure_dirs(root_path)

    # Bypass SSL certificate errors occasionally encountered on university/Windows networks for Zenodo/DFKI hosts
    ssl._create_default_https_context = ssl._create_unverified_context

    try:
        dataset = datasets.EuroSAT(root=str(root_path), download=download)
        image_folder_path = Path(dataset._base_folder) / "2750"
        return image_folder_path, list(dataset.classes)
    except Exception as exc:
        logger.warning(f"Failed to download EuroSAT from default mirror ({exc}). Checking local fallback...")
        candidate = root_path / "eurosat" / "2750"
        if candidate.exists():
            classes = sorted([d.name for d in candidate.iterdir() if d.is_dir()])
            return candidate, classes
        raise RuntimeError(f"Could not download or locate EuroSAT dataset in {root_path}: {exc}") from exc


def generate_synthetic_eurosat_if_needed(
    root_dir: str | Path,
    class_names: List[str],
    samples_per_class: int = 30,
    image_size: int = 64,
    seed: int = 42,
) -> Path:
    """Create a realistic synthetic satellite texture dataset for unit tests or offline environments."""
    rng = np.random.default_rng(seed)
    base_dir = Path(root_dir) / "eurosat" / "2750"

    # Characteristic spectral-spatial signatures (RGB base colors + spatial structures)
    palettes = {
        "AnnualCrop": ([185, 170, 95], "stripes"),
        "Forest": ([35, 85, 40], "canopy"),
        "HerbaceousVegetation": ([110, 155, 80], "patchy"),
        "Highway": ([120, 120, 125], "line"),
        "Industrial": ([175, 170, 180], "blocks"),
        "Pasture": ([85, 145, 65], "smooth"),
        "PermanentCrop": ([135, 130, 75], "grid"),
        "Residential": ([160, 115, 100], "dense_grid"),
        "River": ([55, 105, 135], "meander"),
        "SeaLake": ([25, 65, 125], "water"),
    }

    for cls_name in class_names:
        cls_dir = base_dir / cls_name
        ensure_dirs(cls_dir)
        existing = list(cls_dir.glob("*.jpg"))
        if len(existing) >= samples_per_class:
            continue

        base_rgb, pattern = palettes.get(cls_name, ([128, 128, 128], "smooth"))
        for i in range(samples_per_class):
            img_arr = np.zeros((image_size, image_size, 3), dtype=np.float32)
            for c in range(3):
                img_arr[:, :, c] = base_rgb[c] + rng.normal(0, 12, size=(image_size, image_size))

            yy, xx = np.mgrid[0:image_size, 0:image_size]
            if pattern == "stripes":
                angle = rng.uniform(0, np.pi)
                stripes = 22.0 * np.sin((xx * np.cos(angle) + yy * np.sin(angle)) * 0.45)
                img_arr += stripes[:, :, None]
            elif pattern == "canopy":
                canopy = 18.0 * np.sin(xx * 0.8) * np.cos(yy * 0.8)
                img_arr[:, :, 1] += canopy
            elif pattern == "line":
                pos = rng.integers(16, image_size - 16)
                if rng.random() > 0.5:
                    img_arr[pos - 4 : pos + 4, :, :] = [145, 145, 150]
                else:
                    img_arr[:, pos - 4 : pos + 4, :] = [145, 145, 150]
            elif pattern == "blocks":
                for _ in range(4):
                    r0, c0 = rng.integers(4, image_size - 20, size=2)
                    img_arr[r0 : r0 + 14, c0 : c0 + 16, :] = [215, 210, 220]
            elif pattern == "grid":
                img_arr[::6, :, :] -= 25
                img_arr[:, ::6, :] -= 25
            elif pattern == "dense_grid":
                img_arr[::8, ::8, 0] += 45
                img_arr[2::8, 2::8, :] = [195, 130, 110]
            elif pattern == "meander":
                center = (image_size / 2) + 8.0 * np.sin(xx * 0.15)
                mask = np.abs(yy - center) < 7
                img_arr[mask] = [40, 95, 145]
                img_arr[~mask] = [95, 135, 80]
            elif pattern == "water":
                wave = 6.0 * np.sin(xx * 0.3 + yy * 0.3)
                img_arr[:, :, 2] += wave

            img_uint8 = np.clip(img_arr, 0, 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(cls_dir / f"{cls_name}_{i + 1}.jpg", quality=95)

    return base_dir


def create_stratified_splits(
    config: Dict[str, Any],
    force_recreate: bool = False,
    allow_synthetic_fallback: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Scan dataset directory, perform leakage-free stratified train/val/test splitting, and persist CSVs."""
    data_cfg = config["data"]
    seed = int(config["project"]["seed"])
    splits_dir = Path(data_cfg["splits_dir"])
    ensure_dirs(splits_dir)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    if not force_recreate and train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        if all(Path(p).exists() for p in train_df["filepath"].head(5)):
            logger.info(
                f"Loaded existing stratified splits from {splits_dir} "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})"
            )
            return {
                "train": train_df.to_dict("records"),
                "val": val_df.to_dict("records"),
                "test": test_df.to_dict("records"),
            }

    try:
        image_folder_path, classes = download_or_verify_eurosat(data_cfg["root_dir"], download=True)
    except Exception as exc:
        if not allow_synthetic_fallback:
            raise
        logger.warning(f"Using synthetic satellite fallback due to error: {exc}")
        classes = list(data_cfg["class_names"])
        image_folder_path = generate_synthetic_eurosat_if_needed(
            root_dir=data_cfg["root_dir"],
            class_names=classes,
            samples_per_class=data_cfg.get("max_samples_per_class") or 50,
            image_size=int(data_cfg["image_size"]),
            seed=seed,
        )

    max_per_class = data_cfg.get("max_samples_per_class")
    rng = np.random.default_rng(seed)

    all_records: List[Dict[str, Any]] = []
    for label_idx, cls_name in enumerate(sorted(classes)):
        cls_dir = image_folder_path / cls_name
        files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.tif")))
        if max_per_class is not None and len(files) > int(max_per_class):
            chosen_indices = rng.choice(len(files), size=int(max_per_class), replace=False)
            files = [files[i] for i in sorted(chosen_indices)]
        for fp in files:
            all_records.append({
                "filepath": str(fp.as_posix()),
                "filename": fp.name,
                "class_name": cls_name,
                "label": label_idx,
            })

    df = pd.DataFrame(all_records)
    train_ratio = float(data_cfg["train_ratio"])
    val_ratio = float(data_cfg["val_ratio"])
    test_ratio = float(data_cfg["test_ratio"])

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Step 1: Split train vs (val + test) stratified on label
    train_df, temp_df = train_test_split(
        df,
        test_size=round(1.0 - train_ratio, 4),
        stratify=df["label"],
        random_state=seed,
    )

    # Step 2: Split temp into val vs test stratified on label
    relative_test_ratio = round(test_ratio / (val_ratio + test_ratio), 4)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=seed,
    )

    # Verify zero data leakage across splits
    train_files = set(train_df["filepath"])
    val_files = set(val_df["filepath"])
    test_files = set(test_df["filepath"])
    assert train_files.isdisjoint(val_files), "Data leakage detected between train and val splits!"
    assert train_files.isdisjoint(test_files), "Data leakage detected between train and test splits!"
    assert val_files.isdisjoint(test_files), "Data leakage detected between val and test splits!"

    train_df = train_df.assign(split="train").reset_index(drop=True)
    val_df = val_df.assign(split="val").reset_index(drop=True)
    test_df = test_df.assign(split="test").reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info(
        f"Created stratified splits with zero leakage: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    return {
        "train": train_df.to_dict("records"),
        "val": val_df.to_dict("records"),
        "test": test_df.to_dict("records"),
    }


def build_dataloaders(
    config: Dict[str, Any],
    splits: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Tuple[Dict[str, DataLoader], Dict[str, EuroSATSubsetDataset]]:
    """Construct PyTorch Datasets and DataLoaders for train, val, and test splits."""
    if splits is None:
        splits = create_stratified_splits(config)

    tfms = get_transforms(config)
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"].get("num_workers", 0))
    pin_memory = bool(config["data"].get("pin_memory", True)) and torch.cuda.is_available()

    datasets_dict = {
        "train": EuroSATSubsetDataset(splits["train"], transform=tfms["train"]),
        "val": EuroSATSubsetDataset(splits["val"], transform=tfms["val"]),
        "test": EuroSATSubsetDataset(splits["test"], transform=tfms["test"]),
    }

    loaders_dict = {
        "train": DataLoader(
            datasets_dict["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets_dict["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets_dict["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }

    return loaders_dict, datasets_dict
