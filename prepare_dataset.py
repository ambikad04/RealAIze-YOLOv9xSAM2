import os
import shutil
from pathlib import Path
import random

def prepare_dataset():
    # Source directory containing uploaded images
    source_dir = Path("uploads/headphones")
    
    # Destination directories
    train_dir = Path("headphone_dataset/images/train")
    val_dir = Path("headphone_dataset/images/val")
    train_labels_dir = Path("headphone_dataset/labels/train")
    val_labels_dir = Path("headphone_dataset/labels/val")
    
    # Create label directories if they don't exist
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
    
    if not image_files:
        print("No images found in uploads/headphones directory!")
        return
    
    # Shuffle the files
    random.shuffle(image_files)
    
    # Split into train and validation sets (80% train, 20% val)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Found {len(image_files)} images")
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    
    # Function to create dummy label file (full image as headphone)
    def create_label_file(image_path, label_dir):
        label_path = label_dir / f"{image_path.stem}.txt"
        # Create a label file with the entire image as headphone (class 0)
        with open(label_path, 'w') as f:
            # Format: class x_center y_center width height (all normalized)
            f.write("0 0.5 0.5 1.0 1.0\n")
    
    # Move files and create labels
    for file in train_files:
        shutil.copy2(file, train_dir / file.name)
        create_label_file(file, train_labels_dir)
        
    for file in val_files:
        shutil.copy2(file, val_dir / file.name)
        create_label_file(file, val_labels_dir)
    
    print("\nDataset prepared successfully:")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")

if __name__ == "__main__":
    prepare_dataset() 