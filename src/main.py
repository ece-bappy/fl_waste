"""
Main application for medical waste sorting system.
Integrates detection, federated learning, and robotic control.
"""

import asyncio
import logging
import argparse
import signal
import sys
from pathlib import Path
from typing import Optional

from src.detection import MedicalWasteDetector
from src.federated import FederatedServer, FederatedClient
from src.robotics import SortingController, WasteCategory
from src.utils import ConfigManager

logger = logging.getLogger(__name__)

class MedicalWasteSortingSystem:
    """Main system integrating all components."""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.config = ConfigManager(config_path)
        self.detector = None
        self.robot = None
        self.federated_client = None
        self.federated_server = None
        self.running = False
        
        self._setup_logging()
        self._initialize_components()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging_config = self.config.get_logging_config()
        
        logging.basicConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format=logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(logging_config.get('file', 'logs/system.log')),
                logging.StreamHandler(sys.stdout) if logging_config.get('console_output', True) else logging.NullHandler()
            ]
        )
    
    def _initialize_components(self):
        """Initialize system components."""
        try:
            # Initialize detector
            detection_config = self.config.get_detection_config()
            self.detector = MedicalWasteDetector(
                model_path=detection_config.get('model_path'),
                device=detection_config.get('device', 'cpu')
            )
            logger.info("Detector initialized")
            
            # Initialize robot controller
            robotics_config = self.config.get_robotics_config()
            arduino_config = robotics_config.get('arduino', {})
            
            self.robot = SortingController(
                port=arduino_config.get('port', '/dev/ttyUSB0'),
                baudrate=arduino_config.get('baudrate', 115200),
                timeout=arduino_config.get('timeout', 1.0)
            )
            logger.info("Robot controller initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def start_federated_server(self):
        """Start federated learning server."""
        try:
            federated_config = self.config.get_federated_config()
            server_config = federated_config.get('server', {})
            
            self.federated_server = FederatedServer(
                global_model_path="models/global_model.pt",
                num_clients=server_config.get('min_clients', 2),
                aggregation_rounds=server_config.get('aggregation_rounds', 100),
                server_port=server_config.get('port', 8765)
            )
            
            logger.info("Starting federated learning server...")
            await self.federated_server.start_server()
            
        except Exception as e:
            logger.error(f"Failed to start federated server: {e}")
    
    async def start_federated_client(self, client_id: str):
        """Start federated learning client."""
        try:
            federated_config = self.config.get_federated_config()
            client_config = federated_config.get('client', {})
            server_config = federated_config.get('server', {})
            
            self.federated_client = FederatedClient(
                client_id=client_id,
                server_url=f"ws://{server_config.get('host', 'localhost')}:{server_config.get('port', 8765)}",
                local_data_path=f"data/{client_id}_data.json",
                learning_rate=client_config.get('learning_rate', 0.001)
            )
            
            logger.info(f"Starting federated learning client: {client_id}")
            await self.federated_client.run()
            
        except Exception as e:
            logger.error(f"Failed to start federated client: {e}")
    
    async def process_waste_item(self, image_path: str) -> bool:
        """Process a single waste item through the system."""
        try:
            import cv2
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Detect waste category
            detection_results = self.detector.detect(image, enable_learning=True)
            
            if not detection_results['detections']:
                logger.warning("No waste items detected in image")
                return False
            
            # Get primary category
            categories = detection_results['categories']
            primary_category = max(set(categories), key=categories.count)
            
            # Map to robot waste category
            robot_category = self._map_to_robot_category(primary_category)
            
            # Sort waste item
            success = self.robot.sort_waste(robot_category)
            
            if success:
                logger.info(f"Successfully sorted {primary_category} waste item")
                
                # Update model if needed
                if self.detector.dual_loop_learner:
                    self.detector.dual_loop_learner.slow_loop_learning()
                
                return True
            else:
                logger.error("Failed to sort waste item")
                return False
                
        except Exception as e:
            logger.error(f"Error processing waste item: {e}")
            return False
    
    def _map_to_robot_category(self, detection_category: str) -> WasteCategory:
        """Map detection category to robot waste category."""
        category_mapping = {
            'sharps': WasteCategory.SHARPS,
            'infectious': WasteCategory.INFECTIOUS,
            'pharmaceutical': WasteCategory.PHARMACEUTICAL,
            'pathological': WasteCategory.PATHOLOGICAL,
            'general': WasteCategory.GENERAL,
            'unknown': WasteCategory.GENERAL
        }
        
        return category_mapping.get(detection_category.lower(), WasteCategory.GENERAL)
    
    async def run_simulation(self, num_items: int = 10):
        """Run simulation with synthetic data."""
        logger.info(f"Starting simulation with {num_items} waste items")
        
        # Generate or load test images
        test_images = self._generate_test_images(num_items)
        
        successful_sorts = 0
        
        for i, image_path in enumerate(test_images):
            logger.info(f"Processing item {i+1}/{num_items}")
            
            success = await self.process_waste_item(image_path)
            if success:
                successful_sorts += 1
            
            # Wait for robot to complete
            await asyncio.sleep(2)
        
        logger.info(f"Simulation completed: {successful_sorts}/{num_items} successful sorts")
        return successful_sorts / num_items
    
    def _generate_test_images(self, num_items: int) -> list:
        """Generate test images for simulation."""
        # In a real implementation, this would load actual test images
        # For now, we'll create a placeholder
        test_images = []
        
        for i in range(num_items):
            # Create dummy image path
            image_path = f"test_images/waste_item_{i:03d}.jpg"
            test_images.append(image_path)
        
        return test_images
    
    def stop(self):
        """Stop the system."""
        logger.info("Stopping medical waste sorting system...")
        self.running = False
        
        if self.robot:
            self.robot.disconnect()
        
        if self.federated_client:
            self.federated_client.disconnect()

async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Medical Waste Sorting System")
    parser.add_argument("--mode", choices=["server", "client", "standalone"], 
                       default="standalone", help="System operation mode")
    parser.add_argument("--client-id", type=str, default="client_1", 
                       help="Client ID for federated learning")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                       help="Configuration file path")
    parser.add_argument("--simulate", action="store_true",
                       help="Run simulation mode")
    parser.add_argument("--num-items", type=int, default=10,
                       help="Number of items for simulation")
    
    args = parser.parse_args()
    
    # Create system instance
    system = MedicalWasteSortingSystem(args.config)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.mode == "server":
            logger.info("Starting in server mode...")
            await system.start_federated_server()
            
        elif args.mode == "client":
            logger.info(f"Starting in client mode (ID: {args.client_id})...")
            await system.start_federated_client(args.client_id)
            
        elif args.mode == "standalone":
            logger.info("Starting in standalone mode...")
            
            if args.simulate:
                success_rate = await system.run_simulation(args.num_items)
                logger.info(f"Simulation completed with {success_rate:.2%} success rate")
            else:
                logger.info("Standalone mode ready. Use process_waste_item() method to process items.")
                # Keep system running
                while system.running:
                    await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        system.stop()

if __name__ == "__main__":
    asyncio.run(main())
