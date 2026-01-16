"""
Package initialization for Mental Health AI
"""

from .facial_detector import FacialDetector
from .emotion_detector import EmotionDetector, EmotionCNN
from .stress_detector import StressDetector
from .utils import (
    load_image, save_image, resize_image, create_sample_image,
    draw_text, draw_bar, normalize_image
)

__version__ = "1.0.0"
__author__ = "Mental Health AI Research"

__all__ = [
    "FacialDetector",
    "EmotionDetector",
    "StressDetector",
    "EmotionCNN",
    "load_image",
    "save_image",
    "resize_image",
    "create_sample_image",
    "draw_text",
    "draw_bar",
    "normalize_image"
]
