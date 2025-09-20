"""
Federated learning module for medical waste sorting system.
"""

from .server import FederatedServer
from .client import FederatedClient

__all__ = ['FederatedServer', 'FederatedClient']
