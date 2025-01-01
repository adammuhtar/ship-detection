#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: loader.py
Author: Adam Muhtar <adam.b.muhtar@gmail.com>
Description: PyTorch dataset and data loading utilities for ship classification.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Third party imports
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid


class MASATIDataset(Dataset):
    """Helper class to load the MASATI-v2 dataset."""
    def __init__(
        self,
        data_dir: Path,
        class_map: Dict[str, List[str]],
        transform: transforms.Compose = None
    ) -> None:
        """
        Initialise the MASATIDataset class.
        
        Args:
            data_dir (`Path`): Pathlib Path to MASATI-v2 directory.
            class_map (`Dict[str, List[str]]`): Maps class names to ship/no-ship
                categories.
            transform (`transforms.Compose`): torchvision transforms to apply to
                images.
        """
        self.data_dir = data_dir
        self.class_map = class_map
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Dynamically load image paths and labels
        self._load_images()

    def _load_images(self) -> None:
        """
        Load image paths and labels from the MASATI-v2 dataset directory. The
        function iterates over the class_map and assigns a label to each class.
        The image paths and labels are stored in the `image_paths` and `labels`
        lists, respectively.
        """
        for label, class_dirs in enumerate(self.class_map.values()):
            for class_dir in class_dirs:
                class_path = self.data_dir / class_dir
                if class_path.exists() and class_path.is_dir():
                    try:
                        # Add all image files (.png, .jpg, etc.) to the dataset
                        png_files = list(class_path.glob("*.png"))
                        jpg_files = list(class_path.glob("*.jpg"))
                        jpeg_files = list(class_path.glob("*.jpeg"))
                        
                        self.image_paths.extend(png_files)
                        self.image_paths.extend(jpg_files)
                        self.image_paths.extend(jpeg_files)
                        
                        # Assign the current label to each image
                        self.labels.extend([label] * len(png_files))
                        self.labels.extend([label] * len(jpg_files))
                        self.labels.extend([label] * len(jpeg_files))
                        
                        logging.info(
                            f"Loaded {len(png_files) + len(jpg_files) + len(jpeg_files)} images from {class_path}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Error loading images from {class_path}: {e}"
                        )
                else:
                    logging.warning(
                        f"Directory {class_path} does not exist or is not a directory"
                    )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label


def show_sample_images(
    dataloader: DataLoader, class_names: List[str], n_images: int = 12
):
    """
    Display a grid of sample images with their labels.
    
    Args:
        dataloader (`DataLoader`): PyTorch DataLoader object containing the
            dataset to visualise.
        class_names (`List[str]`): List of class names.
        n_images (`int`): Number of images to display.
    """
    try:
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
    except StopIteration:
        logging.error("The dataloader is empty. Please check the dataset.")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        return

    try:
        # Create a grid of images
        img_grid = make_grid(images[:n_images], nrow=4)
        plt.figure(figsize=(12, 8))
        plt.imshow(img_grid.permute(1, 2, 0))  # Convert to HWC format for visualisation

        # Add labels as text
        labels_list = [class_names[label] for label in labels[:n_images]]
        for i, label_name in enumerate(labels_list):
            plt.text(
                x=50 + i % 4 * 512,
                y=50 + (i // 4) * 512,
                s=label_name,
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.7)
            )

        plt.axis("off")
        plt.title("Sample Images with Labels")
        plt.show()
    except Exception as e:
        logging.error(f"An error occurred while displaying images: {e}")