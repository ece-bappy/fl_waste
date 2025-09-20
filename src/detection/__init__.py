"""
Detection module for medical waste sorting system.
"""

from .yolo_detector import MedicalWasteDetector, DualLoopLearner

__all__ = ['MedicalWasteDetector', 'DualLoopLearner']
