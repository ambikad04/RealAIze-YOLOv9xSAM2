# RealAIze - Smart Retail Object Detection System

This project implements a comprehensive smart retail solution using YOLOv8 for object detection, with features for product recognition, inventory tracking, and real-time monitoring. The system includes both single-object and multi-object detection capabilities, along with a web interface for easy interaction.

## Technologies Used

### Core Technologies
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- CUDA 11.8+ (for GPU acceleration)
- SQLite (for database storage)
- Flask (for web interface)
- NumPy
- Matplotlib
- Ultralytics YOLO

### Models
1. **YOLOv8**
   - Latest version of YOLO architecture
   - Improved accuracy and speed
   - Pre-trained on COCO dataset
   - Custom training support
   - Multiple model sizes (nano, small, medium, large)

## Features
- Real-time object detection using YOLOv8
- Support for multiple product categories
- Web interface for easy interaction
- Database integration for inventory tracking
- Mobile scanning capabilities
- Real-time visualization of detections
- Configurable confidence thresholds
- Support for multiple classes
- GPU acceleration support
- Export results in various formats
- Custom class support
- Multi-threaded processing

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required model weights:
```bash
python download_weights.py
```

## Usage

### Web Interface
Run the web application:
```bash
python app.py
```

### Real-time Detection
Run the real-time detection system:
```bash
python realtime_scan.py  # For real-time scanning
python mobile_scan.py    # For mobile device scanning
```

### Product Detection
```bash
python bottle_detector.py    # For bottle detection
python headphone_detector.py # For headphone detection
python multi_object_detector.py # For multiple object detection
```

### Training
```bash
python train_model.py              # Basic model training
python train_headphone_model.py    # Headphone-specific training
python train_multi_object_model.py # Multi-object training
```

## Project Structure
```
├── app.py                    # Web application
├── smart_retail.py          # Main retail system logic
├── database.py              # Database operations
├── detect.py                # Core detection script
├── realtime_scan.py         # Real-time scanning
├── mobile_scan.py           # Mobile device scanning
├── bottle_detector.py       # Bottle detection
├── headphone_detector.py    # Headphone detection
├── multi_object_detector.py # Multi-object detection
├── models/                  # Model architecture and weights
├── templates/              # Web interface templates
├── uploads/                # Uploaded files directory
├── utils.py                # Utility functions
├── config.py               # Configuration settings
├── download_weights.py     # Script to download model weights
└── requirements.txt        # Project dependencies
```

## Database
The system uses SQLite for data storage, with the following features:
- Product inventory tracking
- Detection history
- User management
- Analytics data

## Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ free disk space for models
- Webcam or mobile device for scanning

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Ultralytics for YOLOv8
- COCO dataset contributors 
