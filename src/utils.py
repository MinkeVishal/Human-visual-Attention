"""
Utility functions for mental health AI project
"""

import numpy as np
import cv2
from typing import Tuple
import torch
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Image array (BGR format)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image to file."""
    cv2.imwrite(output_path, image)
    print(f"✓ Saved image to {output_path}")


def resize_image(image: np.ndarray, max_width: int = 800) -> np.ndarray:
    """
    Resize image to fit within max width while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
    
    Returns:
        Resized image
    """
    if image.shape[1] > max_width:
        ratio = max_width / image.shape[1]
        new_height = int(image.shape[0] * ratio)
        image = cv2.resize(image, (max_width, new_height))
    
    return image


def create_sample_image(shape: Tuple[int, int, int] = (480, 640, 3)) -> np.ndarray:
    """
    Create a synthetic sample image for testing.
    
    Args:
        shape: Image shape (H, W, C)
    
    Returns:
        Random colored image
    """
    return np.random.randint(0, 256, shape, dtype=np.uint8)


def draw_text(image: np.ndarray, text: str, position: Tuple[int, int],
             font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
             thickness: int = 2) -> np.ndarray:
    """
    Draw text on image.
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position
        font_scale: Font size
        color: Text color (BGR)
        thickness: Text thickness
    
    Returns:
        Image with drawn text
    """
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
               font_scale, color, thickness)
    return image


def draw_bar(image: np.ndarray, value: float, label: str,
            position: Tuple[int, int], width: int = 200, height: int = 30) -> np.ndarray:
    """
    Draw a progress bar on image.
    
    Args:
        image: Input image
        value: Value from 0 to 1
        label: Label for bar
        position: (x, y) position
        width: Bar width
        height: Bar height
    
    Returns:
        Image with drawn bar
    """
    x, y = position
    
    # Draw background
    cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
    
    # Draw label
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (255, 255, 255), 1)
    
    # Draw filled bar
    bar_width = int(width * value)
    color = (0, 255 * (1 - value), 255 * value)  # Green to red
    cv2.rectangle(image, (x, y), (x + bar_width, y + height), color, -1)
    
    # Draw border
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 255, 255), 2)
    
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
    
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    return tensor.detach().cpu().numpy()


def numpy_to_tensor(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    return torch.from_numpy(array).to(device)
