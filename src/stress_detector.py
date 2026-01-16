"""
Stress Level Detection Module
Multi-signal stress estimation from facial features and patterns
"""

import numpy as np
from typing import List, Dict, Tuple
import cv2


class StressDetector:
    """
    Detects stress levels from facial features.
    Uses multiple cues: muscle tension, eye patterns, skin color, etc.
    """
    
    STRESS_LEVELS = ['Low', 'Moderate', 'High']
    
    def __init__(self):
        """Initialize stress detector."""
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("✓ Stress Detector initialized")
    
    def detect_stress_level(self, face_image: np.ndarray, 
                           emotion: str = "Neutral") -> Tuple[str, float]:
        """
        Detect stress level from face image.
        
        Args:
            face_image: Face image (H, W, 3)
            emotion: Detected emotion (for context)
        
        Returns:
            stress_level: 'Low', 'Moderate', or 'High'
            stress_score: Score from 0 (low) to 1 (high)
        """
        
        # Extract features
        features = self._extract_stress_features(face_image)
        
        # Emotion-based baseline
        emotion_stress_baseline = {
            'happy': 0.2,
            'sad': 0.5,
            'angry': 0.8,
            'surprised': 0.6,
            'fearful': 0.9,
            'disgusted': 0.7,
            'neutral': 0.4
        }
        
        baseline = emotion_stress_baseline.get(emotion.lower(), 0.5)
        
        # Combine features
        stress_score = baseline
        stress_score += features['muscle_tension'] * 0.2
        stress_score += features['blink_rate'] * 0.15
        stress_score += features['skin_color_intensity'] * 0.15
        
        # Normalize
        stress_score = min(1.0, max(0.0, stress_score / 2.5))
        
        # Classify
        if stress_score < 0.33:
            stress_level = 'Low'
        elif stress_score < 0.66:
            stress_level = 'Moderate'
        else:
            stress_level = 'High'
        
        return stress_level, stress_score
    
    def _extract_stress_features(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Extract stress-related features from face.
        
        Args:
            face_image: Face image
        
        Returns:
            Dictionary of feature values
        """
        
        # 1. Muscle tension (edge intensity)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        muscle_tension = np.sum(edges) / (face_image.shape[0] * face_image.shape[1]) / 255
        
        # 2. Skin color analysis
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        # Extract red channel (stress increases skin redness)
        b, g, r = cv2.split(face_image)
        skin_redness = np.mean(r.astype(float) - g.astype(float)) / 255
        skin_color_intensity = max(0, skin_redness) * 0.5 + 0.3
        
        # 3. Blink rate estimation (simplified - based on eye darkness)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(15, 15))
        blink_rate = min(1.0, len(eyes) / 3)  # Normalize by expected eye count
        
        # 4. Face shape analysis (simplified)
        face_darkness = 1.0 - (np.mean(gray) / 255)  # Darker = more tension
        
        return {
            'muscle_tension': muscle_tension,
            'blink_rate': blink_rate,
            'skin_color_intensity': skin_color_intensity,
            'face_darkness': face_darkness
        }
    
    def detect_stress_batch(self, face_images: List[np.ndarray],
                           emotions: List[str]) -> List[Tuple[str, float]]:
        """
        Detect stress levels for multiple faces.
        
        Args:
            face_images: List of face images
            emotions: List of emotion labels
        
        Returns:
            List of (stress_level, stress_score) tuples
        """
        results = []
        for face, emotion in zip(face_images, emotions):
            stress_level, stress_score = self.detect_stress_level(face, emotion)
            results.append((stress_level, stress_score))
        
        return results
    
    def estimate_mental_health_indicators(self, 
                                         emotions: List[str],
                                         stress_scores: List[float]) -> Dict:
        """
        Estimate mental health indicators from emotion and stress data.
        
        Args:
            emotions: List of detected emotions
            stress_scores: List of stress scores
        
        Returns:
            Dictionary of mental health indicators
        """
        
        if not emotions:
            return {}
        
        # Emotion distribution
        emotion_counts = {}
        for e in emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        # Stress statistics
        avg_stress = np.mean(stress_scores) if stress_scores else 0.5
        max_stress = np.max(stress_scores) if stress_scores else 0.5
        stress_variance = np.var(stress_scores) if len(stress_scores) > 1 else 0
        
        # Mental health assessment
        assessment = {
            'overall_stress': avg_stress,
            'peak_stress': max_stress,
            'stress_stability': 1.0 - stress_variance,  # Lower variance = more stable
            'emotion_distribution': emotion_counts,
            'dominant_emotion': max(emotion_counts, key=emotion_counts.get),
            'emotional_diversity': len(emotion_counts) / len(emotions) if emotions else 0,
            'positive_ratio': (emotion_counts.get('Happy', 0) / len(emotions)) if emotions else 0,
            'negative_ratio': (sum([emotion_counts.get(e, 0) for e in ['Sad', 'Angry', 'Fearful']]) / len(emotions)) if emotions else 0
        }
        
        return assessment
