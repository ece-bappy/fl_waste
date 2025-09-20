"""
Federated learning client for medical waste sorting system.
Handles local training and communication with the central server.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import asyncio
import websockets
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class FederatedClient:
    """Client for federated learning participation."""
    
    def __init__(self, 
                 client_id: str,
                 server_url: str = "ws://localhost:8765",
                 local_data_path: str = None,
                 learning_rate: float = 0.001):
        
        self.client_id = client_id
        self.server_url = server_url
        self.local_data_path = local_data_path
        self.learning_rate = learning_rate
        
        self.local_model = None
        self.global_model = None
        self.websocket = None
        self.training_data = []
        self.is_connected = False
        
        self._initialize_local_model()
        self._load_local_data()
    
    def _initialize_local_model(self):
        """Initialize local model (YOLOv11n)."""
        try:
            from ultralytics import YOLO
            self.local_model = YOLO('yolo11n.pt')
            logger.info(f"Initialized local model for client {self.client_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize local model: {e}")
            raise
    
    def _load_local_data(self):
        """Load local training data."""
        try:
            if self.local_data_path and Path(self.local_data_path).exists():
                # Load local dataset
                self.training_data = self._load_dataset(self.local_data_path)
                logger.info(f"Loaded {len(self.training_data)} training samples")
            else:
                # Create synthetic data for simulation
                self.training_data = self._generate_synthetic_data()
                logger.info(f"Generated {len(self.training_data)} synthetic samples")
                
        except Exception as e:
            logger.error(f"Failed to load local data: {e}")
            self.training_data = self._generate_synthetic_data()
    
    def _load_dataset(self, data_path: str) -> List[Dict]:
        """Load dataset from file."""
        # Implement dataset loading logic
        # This would typically load images and annotations
        return []
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic training data for simulation."""
        synthetic_data = []
        
        # Generate synthetic medical waste images and labels
        waste_types = ['sharps', 'infectious', 'pharmaceutical', 'pathological', 'general']
        
        for i in range(100):  # 100 synthetic samples per client
            synthetic_data.append({
                'image': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
                'label': np.random.choice(waste_types),
                'bbox': np.random.randint(0, 500, (4,)).tolist(),
                'confidence': np.random.uniform(0.7, 1.0)
            })
        
        return synthetic_data
    
    async def connect_to_server(self):
        """Connect to the federated learning server."""
        try:
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            
            # Register with server
            await self._register_with_server()
            
            logger.info(f"Client {self.client_id} connected to server")
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            self.is_connected = False
    
    async def _register_with_server(self):
        """Register this client with the server."""
        registration_message = {
            'type': 'register',
            'client_id': self.client_id,
            'data_size': len(self.training_data),
            'capabilities': ['detection', 'training', 'continual_learning']
        }
        
        await self.websocket.send(json.dumps(registration_message))
    
    async def listen_for_updates(self):
        """Listen for updates from the server."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'global_model':
                    await self._handle_global_model_update(data)
                    
                elif message_type == 'aggregation_request':
                    await self._handle_aggregation_request()
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection to server closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"Error listening for updates: {e}")
    
    async def _handle_global_model_update(self, data: Dict):
        """Handle global model update from server."""
        try:
            # Deserialize global model weights
            global_weights = data.get('weights', {})
            round_number = data.get('round', 0)
            
            # Update local model with global weights
            self._update_local_model_with_global(global_weights)
            
            logger.info(f"Updated local model with global weights from round {round_number}")
            
            # Start local training
            await self._perform_local_training()
            
        except Exception as e:
            logger.error(f"Failed to handle global model update: {e}")
    
    def _update_local_model_with_global(self, global_weights: Dict):
        """Update local model with global weights."""
        try:
            # Convert global weights to tensor format
            state_dict = {}
            for name, param_list in global_weights.items():
                state_dict[name] = torch.tensor(param_list)
            
            # Load global weights into local model
            self.local_model.model.load_state_dict(state_dict)
            
        except Exception as e:
            logger.error(f"Failed to update local model: {e}")
    
    async def _perform_local_training(self):
        """Perform local training on client data."""
        try:
            logger.info(f"Starting local training for client {self.client_id}")
            
            # Train on local data
            training_results = self._train_local_model()
            
            # Prepare model update for server
            model_update = self._prepare_model_update()
            
            # Send update to server
            await self._send_model_update(model_update)
            
            logger.info(f"Local training completed for client {self.client_id}")
            
        except Exception as e:
            logger.error(f"Local training failed: {e}")
    
    def _train_local_model(self) -> Dict:
        """Train the local model on client data."""
        try:
            # Implement local training logic
            # This would involve training the YOLO model on local data
            
            training_loss = 0.0
            num_batches = len(self.training_data) // 32  # Batch size 32
            
            for batch_idx in range(num_batches):
                # Get batch of training data
                batch_data = self.training_data[batch_idx * 32:(batch_idx + 1) * 32]
                
                # Simulate training step
                batch_loss = self._simulate_training_step(batch_data)
                training_loss += batch_loss
            
            avg_loss = training_loss / num_batches if num_batches > 0 else 0.0
            
            return {
                'loss': avg_loss,
                'samples_trained': len(self.training_data),
                'epochs': 1
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'loss': float('inf'), 'samples_trained': 0, 'epochs': 0}
    
    def _simulate_training_step(self, batch_data: List[Dict]) -> float:
        """Simulate a training step (placeholder for actual training)."""
        # In a real implementation, this would:
        # 1. Load batch images
        # 2. Forward pass through model
        # 3. Compute loss
        # 4. Backward pass
        # 5. Update weights
        
        # For simulation, return random loss
        return np.random.uniform(0.1, 0.5)
    
    def _prepare_model_update(self) -> Dict:
        """Prepare model update for sending to server."""
        try:
            # Extract model weights
            model_weights = self._extract_model_weights()
            
            return {
                'client_id': self.client_id,
                'weights': model_weights,
                'data_size': len(self.training_data),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to prepare model update: {e}")
            return {}
    
    def _extract_model_weights(self) -> Dict:
        """Extract weights from local model."""
        try:
            weights = {}
            state_dict = self.local_model.model.state_dict()
            
            for name, param in state_dict.items():
                weights[name] = param.cpu().numpy().tolist()
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to extract model weights: {e}")
            return {}
    
    async def _send_model_update(self, model_update: Dict):
        """Send model update to server."""
        try:
            message = {
                'type': 'model_update',
                'client_id': self.client_id,
                'model_weights': model_update.get('weights', {}),
                'data_size': model_update.get('data_size', 0),
                'timestamp': model_update.get('timestamp', time.time())
            }
            
            await self.websocket.send(json.dumps(message))
            logger.info(f"Sent model update to server")
            
        except Exception as e:
            logger.error(f"Failed to send model update: {e}")
    
    async def _handle_aggregation_request(self):
        """Handle aggregation request from server."""
        try:
            # Prepare and send current model state
            model_update = self._prepare_model_update()
            await self._send_model_update(model_update)
            
        except Exception as e:
            logger.error(f"Failed to handle aggregation request: {e}")
    
    async def run(self):
        """Main client loop."""
        try:
            await self.connect_to_server()
            
            if self.is_connected:
                await self.listen_for_updates()
            
        except Exception as e:
            logger.error(f"Client {self.client_id} failed: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()

async def main():
    """Main function to run a federated learning client."""
    import sys
    
    client_id = sys.argv[1] if len(sys.argv) > 1 else "client_1"
    
    client = FederatedClient(
        client_id=client_id,
        server_url="ws://localhost:8765",
        local_data_path=f"data/{client_id}_data.json"
    )
    
    await client.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
