import os
from pathlib import Path
import shutil
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from tqdm import tqdm

def augment_image(image):
    """Apply data augmentation to the image."""
    augmented = []
    
    # Original image
    augmented.append(image)
    
    # Horizontal flip
    augmented.append(cv2.flip(image, 1))
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    augmented.append(bright)
    
    # Contrast adjustment
    contrast = cv2.convertScaleAbs(image, alpha=1.2, beta=0)
    augmented.append(contrast)
    
    # Slight rotation
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    augmented.append(rotated)
    
    return augmented

def prepare_dataset():
    """Prepare the dataset from uploaded images with augmentation."""
    # Create dataset directories
    dataset_dir = Path("headphone_dataset")
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    labels_dir = dataset_dir / "labels"
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Process both object types
    object_types = ['headphones', 'bottles']
    class_ids = {'headphones': 0, 'bottles': 1}
    
    for object_type in object_types:
        # Get uploaded images for this object type
        uploads_dir = Path(f"uploads/{object_type}")
        if not uploads_dir.exists():
            print(f"No {object_type} directory found. Skipping...")
            continue
            
        image_files = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
        if not image_files:
            print(f"No {object_type} images found. Skipping...")
            continue
        
        # Split images into train and validation sets (80% train, 20% val)
        num_images = len(image_files)
        num_train = int(num_images * 0.8)
        
        print(f"\nPreparing {object_type} dataset with augmentation...")
        # Process training images
        for i, img_path in enumerate(tqdm(image_files[:num_train], desc=f"Processing {object_type} training images")):
            # Read and augment image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            augmented_images = augment_image(image)
            
            # Save augmented images and their labels
            for j, aug_img in enumerate(augmented_images):
                aug_name = f"{object_type}_{img_path.stem}_aug{j}.jpg"
                cv2.imwrite(str(train_dir / aug_name), aug_img)
                
                # Create corresponding label file
                label_path = labels_dir / f"{object_type}_{img_path.stem}_aug{j}.txt"
                with open(label_path, "w") as f:
                    # Format: class_id x_center y_center width height (normalized)
                    f.write(f"{class_ids[object_type]} 0.5 0.5 0.8 0.8\n")
        
        # Process validation images (no augmentation)
        for img_path in tqdm(image_files[num_train:], desc=f"Processing {object_type} validation images"):
            val_name = f"{object_type}_{img_path.name}"
            shutil.copy2(img_path, val_dir / val_name)
            
            # Create corresponding label file
            label_path = labels_dir / f"{object_type}_{img_path.stem}.txt"
            with open(label_path, "w") as f:
                f.write(f"{class_ids[object_type]} 0.5 0.5 0.8 0.8\n")
    
    print(f"\nDataset prepared successfully:")
    print(f"Training images: {len(list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.png')))}")
    print(f"Validation images: {len(list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png')))}")
    return True

def main():
    # Prepare dataset
    if not prepare_dataset():
        return
        
    # Load the base YOLO model
    model = YOLO('yolov8n.pt')
    
    # Train the model with improved parameters
    print("\nStarting model training...")
    results = model.train(
        data='headphone_dataset.yaml',
        epochs=200,                # Increased epochs
        imgsz=640,                # Image size
        batch=8,                  # Smaller batch size for better generalization
        name='multi_object_model',  # New model name
        patience=30,              # Early stopping patience
        save=True,                # Save best model
        device='cuda' if torch.cuda.is_available() else 'cpu',
        pretrained=True,          # Use pretrained weights
        optimizer='Adam',         # Use Adam optimizer
        lr0=0.001,               # Initial learning rate
        lrf=0.01,                # Final learning rate
        momentum=0.937,          # SGD momentum
        weight_decay=0.0005,     # Weight decay
        warmup_epochs=3,         # Warmup epochs
        warmup_momentum=0.8,     # Warmup momentum
        warmup_bias_lr=0.1,      # Warmup bias learning rate
        box=7.5,                 # Box loss gain
        cls=0.5,                 # Class loss gain
        dfl=1.5,                 # Distribution focal loss gain
        close_mosaic=10,         # Disable mosaic for last epochs
        hsv_h=0.015,            # HSV-Hue augmentation
        hsv_s=0.7,              # HSV-Saturation augmentation
        hsv_v=0.4,              # HSV-Value augmentation
        degrees=0.0,             # Rotation augmentation
        translate=0.1,           # Translation augmentation
        scale=0.5,               # Scale augmentation
        shear=0.0,              # Shear augmentation
        perspective=0.0,         # Perspective augmentation
        flipud=0.0,             # Flip up-down augmentation
        fliplr=0.5,             # Flip left-right augmentation
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.0,              # Mixup augmentation
        copy_paste=0.0          # Copy-paste augmentation
    )
    
    print("\nTraining completed!")
    print(f"Best model saved at: {Path('runs/detect/multi_object_model/weights/best.pt')}")
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    print(f"Validation metrics: {metrics}")

if __name__ == "__main__":
    main() 