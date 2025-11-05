import os
from pathlib import Path
import torch
from ultralytics import YOLO

def download_weights():
    """Download required model weights."""
    print("Downloading YOLO model weights...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    try:
        # Download YOLOv8n model (smallest and fastest)
        model = YOLO('yolov8n.pt')
        
        # Save the model
        model_path = models_dir / "best.pt"
        model.save(str(model_path))
        
        print(f"Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    download_weights() 