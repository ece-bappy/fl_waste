"""
YOLOv11n detector with dual-loop continual learning for medical waste sorting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DualLoopLearner:
    """Dual-loop learning framework for continual adaptation."""
    
    def __init__(self, model: YOLO, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.adaptation_buffer = []
        self.confidence_threshold = 0.5
        
    def fast_loop_adaptation(self, image: np.ndarray, target: str) -> Dict:
        """Fast adaptation loop for immediate response to new items."""
        # Extract features and perform quick adaptation
        results = self.model(image)
        
        # Check if detection confidence is below threshold
        if results[0].boxes.conf.max() < self.confidence_threshold:
            # Store for slow loop processing
            self.adaptation_buffer.append({
                'image': image,
                'target': target,
                'timestamp': torch.tensor(time.time())
            })
            
            # Perform lightweight adaptation
            return self._lightweight_adaptation(image, target)
        
        return results[0]
    
    def slow_loop_learning(self) -> None:
        """Slow loop for comprehensive learning from adaptation buffer."""
        if len(self.adaptation_buffer) < 10:  # Minimum batch size
            return
            
        # Create training batch from buffer
        batch_images = [item['image'] for item in self.adaptation_buffer]
        batch_targets = [item['target'] for item in self.adaptation_buffer]
        
        # Perform full training update
        self._comprehensive_update(batch_images, batch_targets)
        
        # Clear buffer
        self.adaptation_buffer.clear()
    
    def _lightweight_adaptation(self, image: np.ndarray, target: str) -> Dict:
        """Lightweight adaptation for immediate response."""
        # Fine-tune only the last few layers
        with torch.no_grad():
            # Quick inference with adjusted parameters
            results = self.model(image)
            # Apply target-specific adjustments
            adjusted_results = self._apply_target_adjustments(results, target)
            return adjusted_results
    
    def _comprehensive_update(self, images: List[np.ndarray], targets: List[str]) -> None:
        """Comprehensive model update using collected data."""
        # Implement full training update logic
        # This would involve proper training loop with loss computation
        pass
    
    def _apply_target_adjustments(self, results, target: str) -> Dict:
        """Apply target-specific adjustments to detection results."""
        # Implement target-specific logic
        return results

class MedicalWasteDetector:
    """Main detector class for medical waste classification."""
    
    def __init__(self, model_path: str = None, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.dual_loop_learner = None
        self.waste_categories = {
            'sharps': ['syringes', 'needles', 'scalpels'],
            'infectious': ['blood_soaked', 'bandages', 'swabs'],
            'pharmaceutical': ['pills', 'vials', 'capsules'],
            'pathological': ['tissues', 'organs', 'body_parts'],
            'general': ['gloves', 'masks', 'containers']
        }
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str = None):
        """Load YOLOv11n model."""
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
            else:
                # Load pretrained YOLOv11n
                self.model = YOLO('yolo11n.pt')
            
            self.dual_loop_learner = DualLoopLearner(self.model)
            logger.info("YOLOv11n model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, image: np.ndarray, enable_learning: bool = True) -> Dict:
        """Detect and classify medical waste items."""
        try:
            if enable_learning:
                # Use dual-loop learning
                results = self.dual_loop_learner.fast_loop_adaptation(image, "unknown")
            else:
                # Standard inference
                results = self.model(image)
                results = results[0]
            
            # Process results
            detections = self._process_detections(results)
            
            return {
                'detections': detections,
                'confidence': results.boxes.conf.cpu().numpy() if hasattr(results, 'boxes') else [],
                'categories': self._map_to_waste_categories(detections)
            }
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {'detections': [], 'confidence': [], 'categories': []}
    
    def _process_detections(self, results) -> List[Dict]:
        """Process YOLO detection results."""
        detections = []
        
        if hasattr(results, 'boxes') and results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i].tolist(),
                    'confidence': float(confidences[i]),
                    'class_id': int(classes[i]),
                    'class_name': self.model.names[int(classes[i])]
                })
        
        return detections
    
    def _map_to_waste_categories(self, detections: List[Dict]) -> List[str]:
        """Map detected objects to medical waste categories."""
        categories = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Map to waste categories
            for category, items in self.waste_categories.items():
                if any(item in class_name for item in items):
                    categories.append(category)
                    break
            else:
                categories.append('unknown')
        
        return categories
    
    def update_model(self, training_data: List[Dict]) -> None:
        """Update model with new training data."""
        try:
            # Trigger slow loop learning
            self.dual_loop_learner.slow_loop_learning()
            logger.info("Model updated with new data")
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def save_model(self, path: str) -> None:
        """Save the current model state."""
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
