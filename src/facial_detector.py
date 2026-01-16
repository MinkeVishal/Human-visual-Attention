"""
Facial Detection & Preprocessing Module
Detects faces in images and prepares them for emotion/stress analysis
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

# dlib is optional - try to import, fall back to OpenCV if not available
try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False

class FacialDetector:
    """
    Detects faces in images using OpenCV and dlib.
    Extracts and preprocesses face regions for emotion analysis.
    """
    
    def __init__(self, use_dlib: bool = False):
        """
        Initialize facial detector.
        
        Args:
            use_dlib: If True, use dlib (more accurate but slower)
                     If False, use OpenCV Cascade (faster, real-time)
        """
        # Force OpenCV if dlib not available
        if use_dlib and not HAS_DLIB:
            print("⚠️ dlib not installed. Using OpenCV instead.")
            self.use_dlib = False
        else:
            self.use_dlib = use_dlib and HAS_DLIB
        
        # OpenCV face cascade
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        # Dlib detector (if available and requested)
        if self.use_dlib:
            self.dlib_detector = dlib.get_frontal_face_detector()
        
        print(f"✓ Facial Detector initialized (Method: {'dlib' if self.use_dlib else 'OpenCV'})")
    
    def detect_faces(self, image: np.ndarray, scale_factor: float = 1.1, 
                    min_neighbors: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            scale_factor: Scale factor for cascade classifier
            min_neighbors: Minimum neighbors for face detection
        
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.use_dlib:
            # dlib detection
            dlib_rects = self.dlib_detector(gray, 1)
            faces = [(rect.left(), rect.top(), rect.width(), rect.height()) 
                    for rect in dlib_rects]
        else:
            # OpenCV cascade detection
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            faces = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        
        return faces
    
    def extract_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                     target_size: Tuple[int, int] = (224, 224),
                     expand_ratio: float = 0.2) -> List[np.ndarray]:
        """
        Extract face regions from image.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            target_size: Target face size (H, W)
            expand_ratio: Expand face region by this ratio for context
        
        Returns:
            List of preprocessed face images
        """
        face_images = []
        
        for (x, y, w, h) in faces:
            # Expand region slightly for context
            expand = int(w * expand_ratio)
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(image.shape[1], x + w + expand)
            y2 = min(image.shape[0], y + h + expand)
            
            face_roi = image[y1:y2, x1:x2]
            
            # Resize to target size
            face_resized = cv2.resize(face_roi, target_size)
            
            face_images.append(face_resized)
        
        return face_images
    
    def draw_faces(self, image: np.ndarray, faces: List[Tuple[int, int, int, int]],
                   emotions: Optional[List[str]] = None,
                   confidences: Optional[List[float]] = None,
                   stress_levels: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw face boxes with emotion and stress labels on image.
        
        Args:
            image: Input image
            faces: List of face bounding boxes
            emotions: List of emotion labels
            confidences: List of confidence scores
            stress_levels: List of stress level labels
        
        Returns:
            Image with drawn faces and labels
        """
        output = image.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label
            if emotions and i < len(emotions):
                emotion = emotions[i]
                conf = confidences[i] if confidences and i < len(confidences) else 0.0
                label = f"{emotion} ({conf:.2f})"
                
                # Emotion color coding
                emotion_colors = {
                    'happy': (0, 255, 0),      # Green
                    'sad': (255, 0, 0),        # Red
                    'angry': (0, 0, 255),      # Blue
                    'surprised': (255, 255, 0),  # Cyan
                    'fearful': (255, 0, 255),  # Magenta
                    'disgusted': (0, 165, 255),  # Orange
                    'neutral': (128, 128, 128)  # Gray
                }
                color = emotion_colors.get(emotion.lower(), (255, 255, 255))
                
                cv2.putText(output, label, (x, y - 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw stress level
            if stress_levels and i < len(stress_levels):
                stress = stress_levels[i]
                stress_colors = {
                    'low': (0, 255, 0),        # Green
                    'moderate': (0, 165, 255),  # Orange
                    'high': (0, 0, 255)        # Red
                }
                color = stress_colors.get(stress.lower(), (255, 255, 255))
                cv2.putText(output, f"Stress: {stress}", (x, y + h + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
    
    def preprocess_face(self, face_image: np.ndarray, 
                       normalize: bool = True) -> np.ndarray:
        """
        Preprocess face image for neural network.
        
        Args:
            face_image: Face image
            normalize: Whether to normalize to [0, 1]
        
        Returns:
            Preprocessed face image
        """
        # Convert to float
        face = face_image.astype(np.float32)
        
        # Normalize
        if normalize:
            face = face / 255.0
        
        return face
