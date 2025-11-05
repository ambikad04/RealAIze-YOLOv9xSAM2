import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
import yaml

def prepare_dataset():
    # Create dataset directories
    dataset_dir = Path("milton_dataset")
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels"
    
    # Clean up existing directories
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    # Create new directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "train").mkdir(parents=True, exist_ok=True)
    (labels_dir / "val").mkdir(parents=True, exist_ok=True)

    # Get all images from uploads directory
    uploads_dir = Path("uploads")
    image_files = list(uploads_dir.glob("upload_*.jpg"))
    
    if not image_files:
        raise FileNotFoundError("No images found in uploads directory. Please upload some images first.")
    
    # Split into train and validation (80% train, 20% validation)
    np.random.shuffle(image_files)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    # Copy files to train and validation directories
    for img_path in train_files:
        shutil.copy2(img_path, train_dir / img_path.name)
        # Create corresponding label file
        create_label_file(labels_dir / "train" / f"{img_path.stem}.txt")
    
    for img_path in val_files:
        shutil.copy2(img_path, val_dir / img_path.name)
        # Create corresponding label file
        create_label_file(labels_dir / "val" / f"{img_path.stem}.txt")

    # Create dataset.yaml file
    dataset_yaml = {
        'path': str(dataset_dir.absolute()),  # dataset root dir
        'train': 'images/train',  # train images (relative to 'path')
        'val': 'images/val',  # val images (relative to 'path')
        'names': {
            0: 'milton_bottle'
        },
        'nc': 1  # number of classes
    }

    # Save dataset.yaml
    with open(dataset_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    print(f"Prepared dataset with {len(train_files)} training images and {len(val_files)} validation images")
    print(f"Dataset configuration saved to {dataset_dir / 'dataset.yaml'}")

def create_label_file(label_path):
    """Create a label file with a single bottle annotation."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    # For now, we'll create a simple label that covers most of the image
    # In a real scenario, you would want to manually annotate the images
    with open(label_path, 'w') as f:
        # Format: class x_center y_center width height
        # All values are normalized to [0,1]
        f.write("0 0.5 0.5 0.8 0.8\n")

def train_model():
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8s.pt')
    
    # Create output directory if it doesn't exist
    output_dir = Path("runs/detect/milton_bottle_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train the model
    results = model.train(
        data='milton_dataset/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='milton_bottle_model',
        patience=50,  # Early stopping patience
        save=True,    # Save best model
        device='cpu',  # Use CPU for training
        project='runs',  # Save in runs directory
        exist_ok=True,  # Overwrite existing files
        verbose=True   # Show detailed progress
    )
    
    print("Training completed!")
    print(f"Model saved to: {results.save_dir}")
    
    # Verify model files exist and copy to a known location
    model_dir = Path(results.save_dir) / "weights"
    if not model_dir.exists():
        print("Error: Model weights directory not found!")
        return
    
    # Create a backup of the model in a known location
    backup_dir = Path("models")
    backup_dir.mkdir(exist_ok=True)
    
    best_model = model_dir / "best.pt"
    last_model = model_dir / "last.pt"
    
    if best_model.exists():
        print(f"Best model found at: {best_model}")
        shutil.copy2(best_model, backup_dir / "best.pt")
        print(f"Copied best model to: {backup_dir / 'best.pt'}")
    elif last_model.exists():
        print(f"Last model found at: {last_model}")
        shutil.copy2(last_model, backup_dir / "last.pt")
        print(f"Copied last model to: {backup_dir / 'last.pt'}")
    else:
        print("Error: No model weights found!")
        return
    
    print("\nModel verification complete!")
    print("You can now run realtime_scan.py to test the model.")

if __name__ == "__main__":
    try:
        print("Preparing dataset...")
        prepare_dataset()
        print("\nTraining model...")
        train_model()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have uploaded images to the 'uploads' directory")
        print("2. Check if the images are in JPG format")
        print("3. Ensure you have enough disk space")
        print("4. Verify that the images contain Milton bottles") 