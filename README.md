# Federated Learning Medical Waste Sorting System

A decentralized, self-improving robotic system for intelligent medical waste sorting using YOLOv11n with federated continual learning.

## Overview

This system addresses the global challenge of medical waste management through:
- Lightweight YOLOv11n detector with dual-loop learning
- Federated learning for privacy-preserving collaborative training
- Cost-effective rotating platform for multi-category sorting
- Real-time adaptation to new waste types

## Repository Structure

```
├── src/
│   ├── detection/          # YOLOv11n detector implementation
│   ├── federated/          # Federated learning framework
│   ├── robotics/           # Robot control and sorting logic
│   └── utils/              # Utility functions
├── arduino/                # Arduino interface code
├── data/                   # Dataset and simulation scripts
├── config/                 # Configuration files
└── tests/                  # Unit tests
```

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Configure Arduino: Upload `arduino/sorting_controller.ino`
3. Run simulation: `python data/simulate_healthcare_nodes.py`
4. Start federated training: `python src/federated/server.py`

## Key Features

- **Privacy-Preserving**: Federated learning keeps data local
- **Adaptive**: Continual learning for new waste types
- **Scalable**: Distributed edge deployment
- **Cost-Effective**: Optimized for real-world deployment
