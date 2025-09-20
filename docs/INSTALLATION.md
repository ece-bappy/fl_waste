# Installation Guide

## Prerequisites

- Python 3.8 or higher
- Arduino IDE (for uploading robot controller code)
- Arduino-compatible microcontroller (Arduino Uno, Mega, or similar)
- Stepper motor and servo motor for robot hardware

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (Intel i5 or equivalent)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and data
- **GPU**: Optional, NVIDIA GPU with CUDA support for acceleration

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Arduino IDE**: Latest version

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/your-org/medical-waste-sorting.git
cd medical-waste-sorting
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Python Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# For development (optional)
pip install -e .[dev]

# For GPU acceleration (optional)
pip install -e .[gpu]
```

### 4. Setup Arduino Hardware

#### Hardware Connections
- **Stepper Motor**: Connect to pins 2 (STEP) and 3 (DIR)
- **Servo Motor**: Connect to pin 5
- **Emergency Stop**: Connect to pin 6 (with pull-up resistor)
- **Position Sensor**: Connect to pin 7
- **Waste Detection**: Connect to pin 8
- **Status LED**: Connect to pin 9

#### Upload Arduino Code
1. Open Arduino IDE
2. Install required libraries:
   - AccelStepper
   - Servo
   - LiquidCrystal_I2C
3. Open `arduino/sorting_controller.ino`
4. Select your Arduino board and port
5. Upload the code

### 5. Configuration

#### Create Configuration File
```bash
# Copy default configuration
cp config/default_config.yaml config/local_config.yaml

# Edit configuration for your setup
# Update Arduino port (Windows: COM3, Linux/Mac: /dev/ttyUSB0)
# Update model paths and other settings
```

#### Setup Data Directories
```bash
# Create necessary directories
mkdir -p models data logs test_images
```

### 6. Download Pre-trained Models
```bash
# Download YOLOv11n model (automatic on first run)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

### 7. Test Installation

#### Test Detection System
```bash
python src/main.py --mode standalone --simulate --num-items 5
```

#### Test Robot Connection
```bash
python -c "
from src.robotics import SortingController
robot = SortingController()
print('Robot Status:', robot.get_status())
robot.disconnect()
"
```

#### Test Federated Learning
```bash
# Terminal 1: Start server
python src/main.py --mode server

# Terminal 2: Start client
python src/main.py --mode client --client-id test_client
```

## Troubleshooting

### Common Issues

#### Arduino Connection Issues
- **Problem**: "Not connected to Arduino"
- **Solution**: Check port configuration in `config/local_config.yaml`
- **Windows**: Use `COM3`, `COM4`, etc.
- **Linux/Mac**: Use `/dev/ttyUSB0`, `/dev/ttyACM0`, etc.

#### Model Loading Issues
- **Problem**: "Failed to load model"
- **Solution**: Ensure internet connection for automatic download
- **Manual**: Download YOLOv11n.pt and place in models/ directory

#### Permission Issues (Linux/Mac)
- **Problem**: "Permission denied" for serial port
- **Solution**: Add user to dialout group:
```bash
sudo usermod -a -G dialout $USER
# Logout and login again
```

#### Memory Issues
- **Problem**: "Out of memory" errors
- **Solution**: Reduce batch size in configuration or use CPU instead of GPU

### Getting Help

1. Check logs in `logs/system.log`
2. Run self-test: `python src/main.py --mode standalone`
3. Verify Arduino connection with Arduino IDE serial monitor
4. Check GitHub issues for known problems

## Next Steps

After successful installation:

1. **Generate Test Data**: Run `python data/simulate_healthcare_nodes.py`
2. **Train Models**: Start federated learning with multiple clients
3. **Deploy System**: Configure for production environment
4. **Monitor Performance**: Use logging and metrics collection

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv

# Remove installed package
pip uninstall medical-waste-sorting
```
