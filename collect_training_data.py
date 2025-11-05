import cv2
import os
from pathlib import Path
import time

def collect_training_data():
    # Create dataset directories
    dataset_dir = Path("milton_dataset")
    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'c' to capture an image")
    print("Press 'q' to quit")
    print("Place the Milton bottle in different positions and angles")
    
    image_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Display the frame
        cv2.imshow('Collect Training Data', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save image
            if image_count < 100:  # Use 80% for training
                save_path = train_dir / f"milton_{image_count:04d}.jpg"
            else:  # Use 20% for validation
                save_path = val_dir / f"milton_{image_count:04d}.jpg"
            
            cv2.imwrite(str(save_path), frame)
            print(f"Saved image {image_count} to {save_path}")
            image_count += 1
            
            # Add small delay to prevent multiple captures
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {image_count} images")
    print(f"Training images: {len(list(train_dir.glob('*.jpg')))}")
    print(f"Validation images: {len(list(val_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    collect_training_data() 