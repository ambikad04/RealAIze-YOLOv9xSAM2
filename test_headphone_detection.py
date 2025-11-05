import cv2
import torch
from ultralytics import YOLO
from database import SmartRetailDB
import time
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# Suppress YOLO's verbose output
os.environ['YOLO_VERBOSE'] = 'False'

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

class HeadphoneDetector:
    def __init__(self):
        # Initialize the model with proper confidence thresholds
        self.model = YOLO('runs/detect/headphone_model2/weights/best.pt')
        self.conf_threshold = 0.35  # Slightly lowered for better recall
        self.max_conf_threshold = 0.60  # Increased to allow more valid detections
        
        # Initialize feature extractor with optimized parameters for your dataset
        self.feature_extractor = cv2.HOGDescriptor(
            (48, 48),    # Increased window size for better feature capture
            (16, 16),    # Increased block size for better feature grouping
            (8, 8),      # Increased block stride for better coverage
            (8, 8),      # Increased cell size for better feature representation
            9            # nbins
        )
        
        # Load reference headphone features
        self.reference_features = self._load_reference_features()
        self.similarity_threshold = 0.52  # Lowered threshold for better matching
        self.min_matches_required = 2  # Require at least 2 matches for better accuracy
        
    def _load_reference_features(self):
        """Load features from all reference headphone images."""
        reference_features = []
        uploads_dir = Path("uploads/headphones")
        
        if not uploads_dir.exists():
            print("No reference headphone images found")
            return []
            
        # Load all reference images
        for img_path in uploads_dir.glob("*.jpg"):
            try:
                # Load and preprocess image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                    
                # Resize to fixed size
                image = cv2.resize(image, (48, 48))  # Increased size for better feature extraction
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization
                equalized = cv2.equalizeHist(gray)
                
                # Apply additional preprocessing
                blurred = cv2.GaussianBlur(equalized, (5, 5), 0)  # Increased blur for noise reduction
                
                # Extract HOG features
                features = self.feature_extractor.compute(blurred)
                reference_features.append(features)
                
            except Exception as e:
                continue
                
        return reference_features

    def _is_valid_headphone(self, image: np.ndarray) -> bool:
        """Check if the detected object matches the reference headphones."""
        if not self.reference_features:
            return False
            
        try:
            # Resize image to fixed size
            image = cv2.resize(image, (48, 48))  # Increased size for better feature extraction
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply additional preprocessing
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)  # Increased blur for noise reduction
            
            # Extract HOG features
            features = self.feature_extractor.compute(blurred)
            
            # Calculate similarity with all reference images
            matches = 0
            for ref_features in self.reference_features:
                similarity = np.dot(features.flatten(), ref_features.flatten()) / (
                    np.linalg.norm(features) * np.linalg.norm(ref_features)
                )
                if similarity >= self.similarity_threshold:
                    matches += 1
            
            # Return true if we have enough good matches
            return matches >= self.min_matches_required
            
        except Exception as e:
            return False

def main():
    # Initialize detector and database
    detector = HeadphoneDetector()
    db = SmartRetailDB()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Tracking variables
    in_cart = False
    last_detection_time = time.time()
    stable_detection_count = 0
    
    print("\n=== Headphone Detection System Started ===")
    print("Press 'q' to quit")
    print("Press 'r' to reset detection")
    print("========================================\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Run detection with proper confidence thresholds
        results = detector.model(frame, conf=detector.conf_threshold, verbose=False)[0]
        
        # Draw detection zone
        height, width = frame.shape[:2]
        margin = 100
        cv2.rectangle(display_frame, (margin, margin), 
                     (width - margin, height - margin), 
                     (0, 255, 255), 2)
        
        # Process detections
        detected = False
        for box in results.boxes:
            # Get box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Skip if confidence is too high (likely false positive)
            if conf > detector.max_conf_threshold:
                continue
            
            # Check if detection is in the zone
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            in_zone = (margin <= center_x <= width - margin and 
                      margin <= center_y <= height - margin)
            
            # Extract the detected object
            detected_object = frame[y1:y2, x1:x2]
            
            # Validate if it's a headphone
            is_headphone = detector._is_valid_headphone(detected_object)
            
            # Draw bounding box with different colors based on validation
            if is_headphone:
                color = (0, 255, 0) if in_zone else (0, 0, 255)
            else:
                color = (0, 0, 255)  # Red for invalid detections
                
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with more information
            label = f"Headphone {conf:.2f}"
            if not in_zone:
                label += " (Move to zone)"
            if not is_headphone:
                label += " (Invalid)"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if in_zone and is_headphone:
                detected = True
                stable_detection_count += 1
                last_detection_time = time.time()
                
                # Add to cart if not already in cart and detection is stable
                if not in_cart and stable_detection_count >= 2:
                    product_id = "HEADPHONE_001"  # HP Gaming Headphone
                    db.add_to_cart(product_id)
                    in_cart = True
                    print(f"\n[{get_timestamp()}] Headphone added to cart!")
                    print(f"Product: HP Gaming Headphone")
                    print(f"Price: ₹1599.00")
                    print(f"Cart Total: ₹1599.00")
            else:
                stable_detection_count = 0
        
        # Remove from cart if not detected for 1 second
        if in_cart and time.time() - last_detection_time > 1.0:
            product_id = "HEADPHONE_001"
            db.remove_from_cart(product_id)
            in_cart = False
            stable_detection_count = 0
            print(f"\n[{get_timestamp()}] Headphone removed from cart!")
            print(f"Product: HP Gaming Headphone")
            print(f"Price: ₹1599.00")
            print(f"Cart Total: ₹0.00")
        
        # Display cart total and detection info
        cart_items = db.get_active_cart()
        total = sum(item['price'] for item in cart_items)
        
        # Add status information
        cv2.putText(display_frame, f"Cart Total: ₹{total:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Detections: {len(results.boxes)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        if not detected:
            cv2.putText(display_frame, "Place headphone in yellow zone", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "Headphone detected!", (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Headphone Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n=== Headphone Detection System Stopped ===")
            break
        elif key == ord('r'):
            # Reset tracking
            in_cart = False
            last_detection_time = time.time()
            print(f"\n[{get_timestamp()}] Detection system reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 