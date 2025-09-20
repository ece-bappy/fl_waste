"""
Simulate healthcare nodes with distinct medical waste datasets.
Creates synthetic data for federated learning validation.
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class WasteItem:
    """Represents a medical waste item."""
    image: np.ndarray
    category: str
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    node_id: str

class HealthcareNodeSimulator:
    """Simulates a healthcare node with specific waste characteristics."""
    
    def __init__(self, node_id: str, specialization: str = "general"):
        self.node_id = node_id
        self.specialization = specialization
        self.waste_items = []
        
        # Node-specific waste distribution
        self.waste_distributions = self._get_specialization_distribution(specialization)
        
    def _get_specialization_distribution(self, specialization: str) -> Dict[str, float]:
        """Get waste category distribution based on node specialization."""
        distributions = {
            "hospital": {
                "sharps": 0.25,
                "infectious": 0.30,
                "pharmaceutical": 0.15,
                "pathological": 0.10,
                "general": 0.20
            },
            "clinic": {
                "sharps": 0.35,
                "infectious": 0.20,
                "pharmaceutical": 0.25,
                "pathological": 0.05,
                "general": 0.15
            },
            "pharmacy": {
                "sharps": 0.10,
                "infectious": 0.05,
                "pharmaceutical": 0.70,
                "pathological": 0.00,
                "general": 0.15
            },
            "dental": {
                "sharps": 0.40,
                "infectious": 0.25,
                "pharmaceutical": 0.10,
                "pathological": 0.00,
                "general": 0.25
            },
            "laboratory": {
                "sharps": 0.20,
                "infectious": 0.40,
                "pharmaceutical": 0.20,
                "pathological": 0.15,
                "general": 0.05
            },
            "general": {
                "sharps": 0.20,
                "infectious": 0.25,
                "pharmaceutical": 0.20,
                "pathological": 0.10,
                "general": 0.25
            }
        }
        
        return distributions.get(specialization, distributions["general"])
    
    def generate_dataset(self, num_samples: int = 1000) -> List[WasteItem]:
        """Generate synthetic dataset for this healthcare node."""
        logger.info(f"Generating {num_samples} samples for {self.node_id} ({self.specialization})")
        
        waste_items = []
        
        for i in range(num_samples):
            # Sample category based on node specialization
            category = self._sample_category()
            
            # Generate synthetic image
            image = self._generate_synthetic_image(category, i)
            
            # Generate bounding box
            bbox = self._generate_bounding_box(image.shape)
            
            # Generate confidence score
            confidence = np.random.uniform(0.7, 0.95)
            
            waste_item = WasteItem(
                image=image,
                category=category,
                bbox=bbox,
                confidence=confidence,
                node_id=self.node_id
            )
            
            waste_items.append(waste_item)
        
        self.waste_items = waste_items
        return waste_items
    
    def _sample_category(self) -> str:
        """Sample waste category based on node distribution."""
        categories = list(self.waste_distributions.keys())
        probabilities = list(self.waste_distributions.values())
        
        return np.random.choice(categories, p=probabilities)
    
    def _generate_synthetic_image(self, category: str, seed: int) -> np.ndarray:
        """Generate synthetic image for waste category."""
        np.random.seed(seed + hash(self.node_id))
        
        # Base image dimensions
        height, width = 640, 640
        
        # Create base image with random background
        image = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)
        
        # Add category-specific visual elements
        if category == "sharps":
            image = self._add_sharps_elements(image)
        elif category == "infectious":
            image = self._add_infectious_elements(image)
        elif category == "pharmaceutical":
            image = self._add_pharmaceutical_elements(image)
        elif category == "pathological":
            image = self._add_pathological_elements(image)
        elif category == "general":
            image = self._add_general_elements(image)
        
        # Add noise and variations
        image = self._add_noise_and_variations(image)
        
        return image
    
    def _add_sharps_elements(self, image: np.ndarray) -> np.ndarray:
        """Add sharp object elements to image."""
        # Draw syringe-like objects
        for _ in range(np.random.randint(1, 4)):
            x = np.random.randint(50, image.shape[1] - 50)
            y = np.random.randint(50, image.shape[0] - 50)
            cv2.rectangle(image, (x, y), (x + 20, y + 80), (100, 100, 100), -1)
            cv2.circle(image, (x + 10, y + 10), 8, (200, 200, 200), -1)
        
        return image
    
    def _add_infectious_elements(self, image: np.ndarray) -> np.ndarray:
        """Add infectious waste elements to image."""
        # Draw blood-soaked materials
        for _ in range(np.random.randint(2, 5)):
            x = np.random.randint(30, image.shape[1] - 100)
            y = np.random.randint(30, image.shape[0] - 100)
            cv2.rectangle(image, (x, y), (x + 60, y + 40), (150, 50, 50), -1)
            cv2.rectangle(image, (x + 10, y + 10), (x + 50, y + 30), (200, 100, 100), -1)
        
        return image
    
    def _add_pharmaceutical_elements(self, image: np.ndarray) -> np.ndarray:
        """Add pharmaceutical elements to image."""
        # Draw pill bottles and containers
        for _ in range(np.random.randint(1, 3)):
            x = np.random.randint(50, image.shape[1] - 80)
            y = np.random.randint(50, image.shape[0] - 120)
            cv2.rectangle(image, (x, y), (x + 40, y + 80), (150, 150, 200), -1)
            cv2.rectangle(image, (x + 5, y + 5), (x + 35, y + 15), (200, 200, 255), -1)
        
        return image
    
    def _add_pathological_elements(self, image: np.ndarray) -> np.ndarray:
        """Add pathological waste elements to image."""
        # Draw tissue-like shapes
        for _ in range(np.random.randint(1, 3)):
            x = np.random.randint(40, image.shape[1] - 120)
            y = np.random.randint(40, image.shape[0] - 120)
            points = np.array([[x, y], [x + 80, y + 20], [x + 100, y + 60], [x + 20, y + 80]], np.int32)
            cv2.fillPoly(image, [points], (180, 120, 120))
        
        return image
    
    def _add_general_elements(self, image: np.ndarray) -> np.ndarray:
        """Add general waste elements to image."""
        # Draw various general items
        for _ in range(np.random.randint(2, 6)):
            x = np.random.randint(30, image.shape[1] - 60)
            y = np.random.randint(30, image.shape[0] - 60)
            cv2.rectangle(image, (x, y), (x + 40, y + 40), (120, 120, 120), -1)
        
        return image
    
    def _add_noise_and_variations(self, image: np.ndarray) -> np.ndarray:
        """Add noise and variations to make images more realistic."""
        # Add Gaussian noise
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random blur
        if np.random.random() < 0.3:
            kernel_size = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        return image
    
    def _generate_bounding_box(self, image_shape: Tuple[int, int, int]) -> List[int]:
        """Generate random bounding box for waste item."""
        height, width = image_shape[:2]
        
        # Random object size (10-30% of image)
        obj_width = np.random.randint(width // 10, width // 3)
        obj_height = np.random.randint(height // 10, height // 3)
        
        # Random position
        x = np.random.randint(0, width - obj_width)
        y = np.random.randint(0, height - obj_height)
        
        return [x, y, obj_width, obj_height]
    
    def save_dataset(self, output_dir: str):
        """Save generated dataset to files."""
        output_path = Path(output_dir) / self.node_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save images and annotations
        annotations = []
        
        for i, waste_item in enumerate(self.waste_items):
            # Save image
            image_path = output_path / f"image_{i:04d}.jpg"
            cv2.imwrite(str(image_path), waste_item.image)
            
            # Prepare annotation
            annotation = {
                "image_id": i,
                "image_path": str(image_path.relative_to(output_path)),
                "category": waste_item.category,
                "bbox": waste_item.bbox,
                "confidence": waste_item.confidence,
                "node_id": waste_item.node_id
            }
            annotations.append(annotation)
        
        # Save annotations file
        annotations_path = output_path / "annotations.json"
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Save dataset summary
        summary = {
            "node_id": self.node_id,
            "specialization": self.specialization,
            "total_samples": len(self.waste_items),
            "category_distribution": self._get_category_counts(),
            "waste_distributions": self.waste_distributions
        }
        
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved {len(self.waste_items)} samples for {self.node_id}")
    
    def _get_category_counts(self) -> Dict[str, int]:
        """Get count of items per category."""
        counts = {}
        for waste_item in self.waste_items:
            counts[waste_item.category] = counts.get(waste_item.category, 0) + 1
        return counts

def create_healthcare_network(num_nodes: int = 5, samples_per_node: int = 1000) -> List[HealthcareNodeSimulator]:
    """Create a network of healthcare nodes with different specializations."""
    
    specializations = ["hospital", "clinic", "pharmacy", "dental", "laboratory"]
    nodes = []
    
    logger.info(f"Creating {num_nodes} healthcare nodes...")
    
    for i in range(num_nodes):
        specialization = specializations[i % len(specializations)]
        node_id = f"node_{i+1}_{specialization}"
        
        node = HealthcareNodeSimulator(node_id, specialization)
        node.generate_dataset(samples_per_node)
        nodes.append(node)
    
    return nodes

def main():
    """Main function to generate healthcare network datasets."""
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = "data/healthcare_nodes"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate healthcare network
    nodes = create_healthcare_network(num_nodes=5, samples_per_node=500)
    
    # Save all datasets
    for node in nodes:
        node.save_dataset(output_dir)
    
    # Create network summary
    network_summary = {
        "total_nodes": len(nodes),
        "total_samples": sum(len(node.waste_items) for node in nodes),
        "nodes": [
            {
                "node_id": node.node_id,
                "specialization": node.specialization,
                "samples": len(node.waste_items),
                "distribution": node.waste_distributions
            }
            for node in nodes
        ]
    }
    
    summary_path = Path(output_dir) / "network_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(network_summary, f, indent=2)
    
    logger.info(f"Healthcare network simulation completed!")
    logger.info(f"Total nodes: {network_summary['total_nodes']}")
    logger.info(f"Total samples: {network_summary['total_samples']}")
    logger.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
