"""
Configuration management for medical waste sorting system.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for the medical waste sorting system."""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.config_path = Path(config_path)
        self.config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if self.config_path.suffix == '.yaml' or self.config_path.suffix == '.yml':
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            elif self.config_path.suffix == '.json':
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                logger.error(f"Unsupported config file format: {self.config_path.suffix}")
                self.config = self._get_default_config()
                
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "detection": {
                "model_path": "models/yolo11n.pt",
                "confidence_threshold": 0.5,
                "device": "cpu",
                "dual_loop_learning": {
                    "enabled": True,
                    "fast_loop_threshold": 0.5,
                    "slow_loop_batch_size": 10
                }
            },
            "federated_learning": {
                "server": {
                    "host": "localhost",
                    "port": 8765,
                    "aggregation_rounds": 100,
                    "min_clients": 2
                },
                "client": {
                    "learning_rate": 0.001,
                    "local_epochs": 1,
                    "batch_size": 32
                }
            },
            "robotics": {
                "arduino": {
                    "port": "/dev/ttyUSB0",
                    "baudrate": 115200,
                    "timeout": 1.0
                },
                "sorting": {
                    "timeout": 30.0,
                    "retry_attempts": 3
                }
            },
            "data": {
                "dataset_path": "data/healthcare_nodes",
                "cache_size": 1000,
                "augmentation": {
                    "enabled": True,
                    "rotation_range": 15,
                    "brightness_range": 0.2
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/system.log"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """Save configuration to file."""
        save_path = Path(output_path) if output_path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
                with open(save_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif save_path.suffix == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_detection_config(self) -> Dict[str, Any]:
        """Get detection-specific configuration."""
        return self.get('detection', {})
    
    def get_federated_config(self) -> Dict[str, Any]:
        """Get federated learning configuration."""
        return self.get('federated_learning', {})
    
    def get_robotics_config(self) -> Dict[str, Any]:
        """Get robotics configuration."""
        return self.get('robotics', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
