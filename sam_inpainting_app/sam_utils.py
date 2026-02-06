"""
sam_utils.py

Utilities for:
- Loading images
- Running Meta's Segment Anything Model (SAM)
- Creating, expanding, and saving masks
"""

import os
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


# ---------------------------------------------------
# Model Initialization
# ---------------------------------------------------

def initialize_sam_model(
    model_type: str = "vit_h",
    checkpoint_path: str = "models/sam_vit_h_4b8939.pth"
) -> SamPredictor:
    """
    Load SAM model and return predictor.

    Args:
        model_type: SAM backbone type.
        checkpoint_path: Path to SAM weight file.

    Returns:
        SamPredictor object.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("SAM checkpoint not found!")

    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam_model.to(device)

    return SamPredictor(sam_model)


# ---------------------------------------------------
# Image Utilities
# ---------------------------------------------------

def load_image(image_path: str) -> np.ndarray:
    """
    Load image in RGB format.

    Args:
        image_path: Path to image file.

    Returns:
        RGB image array.
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop_image(image_path: str, x: int, y: int, crop_size: int = 512) -> np.ndarray:
    """
    Crop a 512×512 region centered around (x, y).

    Ensures region stays within image boundaries.
    """
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    start_x = max(0, x - crop_size // 2)
    start_y = max(0, y - crop_size // 2)
    end_x = min(w, start_x + crop_size)
    end_y = min(h, start_y + crop_size)

    start_x = max(0, end_x - crop_size)
    start_y = max(0, end_y - crop_size)

    return image[start_y:end_y, start_x:end_x]


# ---------------------------------------------------
# Mask Generation
# ---------------------------------------------------

def generate_mask(image: np.ndarray, predictor: SamPredictor, points: list) -> np.ndarray:
    """
    Generate SAM mask from user click points.

    Args:
        image: RGB image.
        predictor: SAM predictor.
        points: List of (x, y) coordinates.

    Returns:
        Best scoring mask.
    """
    predictor.set_image(image)

    input_points = np.array(points)
    input_labels = np.ones(len(points))

    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    return masks[np.argmax(scores)]


def expand_and_feather_mask(
    mask: np.ndarray,
    dilation_iterations: int = 10,
    blur_kernel_size: int = 21
) -> np.ndarray:
    """
    Expand mask and feather edges for smoother blending.
    """
    kernel = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilation_iterations)
    feathered = cv2.GaussianBlur(expanded.astype(np.float32),
                                 (blur_kernel_size, blur_kernel_size), 0)

    return np.clip(feathered, 0, 1)


def save_mask(mask: np.ndarray, output_path: str):
    """Save mask as 8-bit image."""
    cv2.imwrite(output_path, (mask * 255).astype(np.uint8))
