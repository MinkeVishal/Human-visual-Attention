"""
Emotion Detection Module
CNN-based facial emotion recognition (7 categories)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")


class EmotionCNN(nn.Module):
    """
    Convolutional Neural Network for facial emotion recognition.
    7-class emotion classification: Happy, Sad, Angry, Surprised, Fearful, Disgusted, Neutral
    """
    
    def __init__(self, num_classes: int = 7):
        super(EmotionCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        # Global pooling & FC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EmotionDetector:
    """
    Facial emotion detection using CNN.
    Classifies faces into 7 emotions.
    """
    
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
    
    def __init__(self, device: str = "cpu", pretrained: bool = True):
        """
        Initialize emotion detector.
        
        Args:
            device: 'cpu' or 'cuda'
            pretrained: If True, use pretrained weights (simulated)
        """
        self.device = torch.device(device)
        self.model = EmotionCNN(num_classes=len(self.EMOTIONS))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Emotion Detector initialized on {device}")
    
    def predict(self, face_images: List[np.ndarray]) -> Tuple[List[str], List[float]]:
        """
        Predict emotions for face images.
        
        Args:
            face_images: List of face images (normalized 0-1)
        
        Returns:
            emotions: List of emotion labels
            confidences: List of confidence scores
        """
        if not face_images:
            return [], []
        
        emotions = []
        confidences = []
        
        with torch.no_grad():
            for face in face_images:
                # Prepare image
                if face.ndim == 2:  # Grayscale
                    face = np.stack([face] * 3, axis=-1)
                
                # Convert to tensor
                face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
                face_tensor = face_tensor.to(self.device).float()
                
                # Forward pass
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_idx = predicted.item()
                emotion = self.EMOTIONS[emotion_idx]
                conf = confidence.item()
                
                emotions.append(emotion)
                confidences.append(conf)
        
        return emotions, confidences
    
    def predict_distribution(self, face_images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get full emotion probability distribution.
        
        Args:
            face_images: List of face images
        
        Returns:
            List of emotion probability distributions
        """
        distributions = []
        
        with torch.no_grad():
            for face in face_images:
                if face.ndim == 2:
                    face = np.stack([face] * 3, axis=-1)
                
                face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
                face_tensor = face_tensor.to(self.device).float()
                
                outputs = self.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                dist = probabilities.cpu().numpy()[0]
                
                distributions.append(dist)
        
        return distributions
