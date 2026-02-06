"""
blending_utils.py

Performs Poisson blending between images.
"""

import cv2
import numpy as np
from skimage import exposure


def poisson_blend(background, foreground, mask, position):
    """Blend foreground into background using seamless cloning."""
    x, y = position
    center = (x + foreground.shape[1] // 2, y + foreground.shape[0] // 2)
    return cv2.seamlessClone(foreground, background, mask, center, cv2.NORMAL_CLONE)


def blend_images(bg_path, fg_path, mask_path, output_path, position=(200, 200)):
    """Full blending pipeline."""
    background = cv2.imread(bg_path)
    foreground = cv2.imread(fg_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask = cv2.resize(mask, (foreground.shape[1], foreground.shape[0]))
    matched_fg = exposure.match_histograms(foreground, background, channel_axis=-1)

    blended = poisson_blend(background, matched_fg, mask, position)
    cv2.imwrite(output_path, blended)
