"""
Federated learning server for medical waste sorting system.
Coordinates global model training across distributed healthcare nodes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import asyncio
import websockets
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class FederatedServer:
    """Central server for federated learning coordination."""
    
    def __init__(self, 
                 global_model_path: str,
                 num_clients: int = 5,
                 aggregation_rounds: int = 100,
                 server_port: int = 8765):
        
        self.global_model_path = global_model_path
        self.num_clients = num_clients
        self.aggregation_rounds = aggregation_rounds
        self.server_port = server_port
        
        self.global_model = None
        self.client_models = {}
        self.aggregation_weights = {}
        self.round_completed = 0
        
        self._initialize_global_model()
    
    def _initialize_global_model(self):
        """Initialize the global model."""
        try:
            if Path(self.global_model_path).exists():
                self.global_model = torch.load(self.global_model_path)
                logger.info("Loaded existing global model")
            else:
                # Initialize with YOLOv11n weights
                from ultralytics import YOLO
                self.global_model = YOLO('yolo11n.pt')
                logger.info("Initialized new global model")
                
        except Exception as e:
            logger.error(f"Failed to initialize global model: {e}")
            raise
    
    async def start_server(self):
        """Start the federated learning server."""
        logger.info(f"Starting federated learning server on port {self.server_port}")
        
        async with websockets.serve(self.handle_client, "localhost", self.server_port):
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket, path):
        """Handle client connections and federated learning rounds."""
        client_id = None
        
        try:
            async for message in websocket:
                data = json.loads(message)
                message_type = data.get('type')
                
                if message_type == 'register':
                    client_id = await self._register_client(websocket, data)
                    
                elif message_type == 'model_update':
                    await self._receive_model_update(client_id, data)
                    
                elif message_type == 'request_global_model':
                    await self._send_global_model(websocket)
                    
                elif message_type == 'aggregate':
                    await self._trigger_aggregation()
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
    
    async def _register_client(self, websocket, data: Dict) -> str:
        """Register a new client."""
        client_id = data.get('client_id', f'client_{len(self.client_models)}')
        self.client_models[client_id] = {
            'websocket': websocket,
            'last_update': time.time(),
            'model_updates': 0,
            'data_size': data.get('data_size', 0)
        }
        
        # Send initial global model
        await self._send_global_model(websocket)
        
        logger.info(f"Registered client: {client_id}")
        return client_id
    
    async def _receive_model_update(self, client_id: str, data: Dict):
        """Receive model updates from clients."""
        if client_id not in self.client_models:
            return
        
        # Store client model update
        model_weights = data.get('model_weights')
        if model_weights:
            # Convert to tensor format
            self.client_models[client_id]['weights'] = self._deserialize_weights(model_weights)
            self.client_models[client_id]['last_update'] = time.time()
            self.client_models[client_id]['model_updates'] += 1
            
            logger.info(f"Received model update from {client_id}")
    
    async def _send_global_model(self, websocket):
        """Send global model to client."""
        try:
            # Serialize global model weights
            serialized_weights = self._serialize_weights(self.global_model)
            
            message = {
                'type': 'global_model',
                'weights': serialized_weights,
                'round': self.round_completed
            }
            
            await websocket.send(json.dumps(message))
            logger.info("Sent global model to client")
            
        except Exception as e:
            logger.error(f"Failed to send global model: {e}")
    
    async def _trigger_aggregation(self):
        """Trigger federated aggregation of client models."""
        if len(self.client_models) < 2:
            logger.warning("Not enough clients for aggregation")
            return
        
        try:
            # Collect model weights from clients
            client_weights = []
            weights = []
            
            for client_id, client_data in self.client_models.items():
                if 'weights' in client_data:
                    client_weights.append(client_data['weights'])
                    # Use data size as weight
                    weights.append(client_data['data_size'])
            
            if len(client_weights) == 0:
                logger.warning("No client weights available for aggregation")
                return
            
            # Perform federated averaging
            self._federated_averaging(client_weights, weights)
            
            # Save updated global model
            self._save_global_model()
            
            self.round_completed += 1
            logger.info(f"Aggregation round {self.round_completed} completed")
            
            # Broadcast updated model to all clients
            await self._broadcast_global_model()
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
    
    def _federated_averaging(self, client_weights: List[Dict], weights: List[int]):
        """Perform federated averaging of client model weights."""
        if not client_weights:
            return
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get all parameter names from first client
        param_names = list(client_weights[0].keys())
        
        for param_name in param_names:
            # Weighted average of parameter
            weighted_sum = torch.zeros_like(client_weights[0][param_name])
            
            for i, client_weight in enumerate(client_weights):
                if param_name in client_weight:
                    weighted_sum += normalized_weights[i] * client_weight[param_name]
            
            aggregated_weights[param_name] = weighted_sum
        
        # Update global model with aggregated weights
        self._update_global_model(aggregated_weights)
    
    def _update_global_model(self, aggregated_weights: Dict):
        """Update global model with aggregated weights."""
        try:
            # Load aggregated weights into global model
            state_dict = self.global_model.model.state_dict()
            
            for param_name, param_value in aggregated_weights.items():
                if param_name in state_dict:
                    state_dict[param_name] = param_value
            
            self.global_model.model.load_state_dict(state_dict)
            
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
    
    def _serialize_weights(self, model) -> Dict:
        """Serialize model weights for transmission."""
        try:
            weights = {}
            state_dict = model.model.state_dict()
            
            for name, param in state_dict.items():
                weights[name] = param.cpu().numpy().tolist()
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to serialize weights: {e}")
            return {}
    
    def _deserialize_weights(self, serialized_weights: Dict) -> Dict:
        """Deserialize weights from client updates."""
        try:
            weights = {}
            
            for name, param_list in serialized_weights.items():
                weights[name] = torch.tensor(param_list)
            
            return weights
            
        except Exception as e:
            logger.error(f"Failed to deserialize weights: {e}")
            return {}
    
    def _save_global_model(self):
        """Save the global model."""
        try:
            torch.save(self.global_model, self.global_model_path)
            logger.info(f"Global model saved to {self.global_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save global model: {e}")
    
    async def _broadcast_global_model(self):
        """Broadcast updated global model to all connected clients."""
        serialized_weights = self._serialize_weights(self.global_model)
        
        message = {
            'type': 'global_model',
            'weights': serialized_weights,
            'round': self.round_completed
        }
        
        for client_id, client_data in self.client_models.items():
            try:
                websocket = client_data['websocket']
                await websocket.send(json.dumps(message))
                
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")

async def main():
    """Main function to run the federated learning server."""
    server = FederatedServer(
        global_model_path='models/global_model.pt',
        num_clients=5,
        aggregation_rounds=100
    )
    
    await server.start_server()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
