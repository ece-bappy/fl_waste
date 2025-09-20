"""
Tests for detection module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detection import MedicalWasteDetector, DualLoopLearner

class TestMedicalWasteDetector:
    """Test cases for MedicalWasteDetector."""
    
    @patch('detection.yolo_detector.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test detector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = MedicalWasteDetector()
        
        assert detector.model == mock_model
        assert detector.device == 'cpu'
        assert detector.waste_categories is not None
        mock_yolo.assert_called_once_with('yolo11n.pt')
    
    @patch('detection.yolo_detector.YOLO')
    def test_detector_with_custom_path(self, mock_yolo):
        """Test detector initialization with custom model path."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = MedicalWasteDetector(model_path="custom_model.pt")
        
        mock_yolo.assert_called_once_with('custom_model.pt')
    
    @patch('detection.yolo_detector.YOLO')
    def test_detect_method(self, mock_yolo):
        """Test waste detection method."""
        # Setup mock
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.conf = Mock()
        mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
        mock_result.boxes.cls = Mock()
        mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
        mock_result.boxes.xyxy = Mock()
        mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]])
        mock_model.names = {0: 'syringe'}
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = MedicalWasteDetector()
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test detection
        results = detector.detect(test_image)
        
        assert 'detections' in results
        assert 'confidence' in results
        assert 'categories' in results
        assert len(results['detections']) == 1
        assert results['detections'][0]['confidence'] == 0.8
        assert results['detections'][0]['class_name'] == 'syringe'

class TestDualLoopLearner:
    """Test cases for DualLoopLearner."""
    
    def test_dual_loop_learner_initialization(self):
        """Test dual loop learner initialization."""
        mock_model = Mock()
        learner = DualLoopLearner(mock_model, learning_rate=0.001)
        
        assert learner.model == mock_model
        assert learner.learning_rate == 0.001
        assert learner.confidence_threshold == 0.5
        assert len(learner.adaptation_buffer) == 0
    
    def test_fast_loop_adaptation_high_confidence(self):
        """Test fast loop adaptation with high confidence detection."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.conf = Mock()
        mock_result.boxes.conf.max.return_value = 0.8  # High confidence
        mock_model.return_value = [mock_result]
        
        learner = DualLoopLearner(mock_model)
        
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = learner.fast_loop_adaptation(test_image, "syringe")
        
        assert result == mock_result
        assert len(learner.adaptation_buffer) == 0  # Should not buffer high confidence
    
    def test_fast_loop_adaptation_low_confidence(self):
        """Test fast loop adaptation with low confidence detection."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = Mock()
        mock_result.boxes.conf = Mock()
        mock_result.boxes.conf.max.return_value = 0.3  # Low confidence
        mock_model.return_value = [mock_result]
        
        learner = DualLoopLearner(mock_model)
        
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = learner.fast_loop_adaptation(test_image, "unknown")
        
        assert len(learner.adaptation_buffer) == 1  # Should buffer low confidence
        assert learner.adaptation_buffer[0]['image'].shape == test_image.shape
        assert learner.adaptation_buffer[0]['target'] == "unknown"

if __name__ == "__main__":
    pytest.main([__file__])
