import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from ultralytics import YOLO
import supervision as sv
from database import SmartRetailDB
import time
from pathlib import Path
import os

class BottleDetector:
    def __init__(
        self,
        model_path: str,
        db_path: str = "smart_retail.db",
        conf_threshold: float = 0.36,  # Minimum confidence threshold of 36%
        max_conf_threshold: float = 0.55,  # Maximum confidence threshold of 55%
        iou_threshold: float = 0.45,    # Balanced IoU threshold
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.max_conf_threshold = max_conf_threshold
        self.iou_threshold = iou_threshold
        self.db = SmartRetailDB(db_path)
        self.device = device
        
        # Print model information
        print(f"Model loaded successfully")
        print(f"Model classes: {self.model.names}")
        print(f"Confidence threshold range: {self.conf_threshold:.2f} - {self.max_conf_threshold:.2f}")
        print(f"IoU threshold: {self.iou_threshold}")
        
        # Tracking variables
        self.tracked_objects = {}  # {track_id: {product_id, last_seen, position, in_frame, in_cart}}
        self.pickup_threshold = 0.35  # Reduced threshold for pickup detection
        self.putback_threshold = 0.35  # Reduced threshold for putback detection
        self.last_positions = {}  # {track_id: last_position}
        self.frame_count = 0
        self.missing_frames_threshold = 2  # Reduced for faster removal detection
        self.stable_frames_threshold = 4  # Balanced for stable detection
        self.stable_frames = {}  # {track_id: stable_frame_count}
        self.detection_history = {}  # {track_id: [positions]}
        self.max_history_size = 5  # Reduced history size for faster response
        self.confidence_history = {}  # {track_id: [confidences]}
        self.debug_info = {}  # Store debug information
        self.uploaded_bottle_features = None  # Store features of uploaded bottle
        self.similarity_threshold = 0.75  # Increased threshold for stricter matching
        self.last_cart_state = set()  # Track last cart state for change detection
        self.removal_timeout = 1.0  # Reduced timeout for faster removal detection
        self.min_detection_size = 80  # Reduced minimum size for better detection
        self.aspect_ratio_range = (0.8, 5.0)  # More lenient aspect ratio range for bottles
        
        # Detection zone settings
        self.detection_zone = {
            'enabled': True,
            'margin': 100,  # Increased margin for better zone
            'color': (0, 255, 255),  # Yellow color for the zone
            'thickness': 2,
            'alpha': 0.3  # Transparency of the zone
        }
        
        # Stability tracking
        self.stability_threshold = 20  # Increased pixels of movement allowed
        self.min_confidence_history = 3  # Reduced minimum number of high confidence detections
        self.min_confidence = 0.35  # Reduced minimum confidence for stable detection
        
        # Feature extraction settings
        self.feature_extractor = cv2.HOGDescriptor(
            (224, 224),  # winSize
            (16, 16),    # blockSize
            (8, 8),      # blockStride
            (8, 8),      # cellSize
            9            # nbins
        )
        
        # Load reference headphone features
        self.reference_features = self._load_reference_headphone_features()
        self.min_matches_required = 3  # Minimum number of reference images that must match

    def _load_reference_headphone_features(self):
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
                    print(f"Failed to load image: {img_path}")
                    continue
                    
                # Resize to fixed size
                image = cv2.resize(image, (224, 224))
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply histogram equalization
                equalized = cv2.equalizeHist(gray)
                
                # Apply additional preprocessing
                blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                
                # Extract HOG features
                features = self.feature_extractor.compute(thresh)
                reference_features.append(features)
                
            except Exception as e:
                print(f"Error processing reference image {img_path}: {str(e)}")
                continue
                
        print(f"Loaded {len(reference_features)} reference headphone features")
        return reference_features

    def _load_uploaded_bottle_features(self):
        """Load features from the uploaded bottle image."""
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            print("No uploads directory found")
            return
            
        # Find the most recent image in uploads
        image_files = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
        if not image_files:
            print("No bottle images found in uploads directory")
            return
            
        latest_image = max(image_files, key=lambda x: x.stat().st_mtime)
        print(f"Loading features from: {latest_image}")
        
        try:
            # Load and preprocess image
            image = cv2.imread(str(latest_image))
            if image is None:
                print(f"Failed to load image: {latest_image}")
                return
                
            # Resize to fixed size
            image = cv2.resize(image, (224, 224))
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply additional preprocessing
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Extract HOG features
            features = self.feature_extractor.compute(thresh)
            
            # Store features
            self.uploaded_bottle_features = features
            print("Successfully loaded uploaded bottle features")
            
        except Exception as e:
            print(f"Error loading uploaded bottle features: {str(e)}")

    def _draw_detection_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the detection zone on the frame."""
        height, width = frame.shape[:2]
        margin = self.detection_zone['margin']
        
        # Calculate zone coordinates
        x1 = margin
        y1 = margin
        x2 = width - margin
        y2 = height - margin
        
        # Create overlay for semi-transparent zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.detection_zone['color'], -1)
        cv2.addWeighted(overlay, self.detection_zone['alpha'], frame, 1 - self.detection_zone['alpha'], 0, frame)
        
        # Draw zone border
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.detection_zone['color'], self.detection_zone['thickness'])
        
        # Add text
        cv2.putText(frame, "Place Bottle Here", (x1 + 10, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.detection_zone['color'], 2)
        
        return frame

    def _is_in_detection_zone(self, x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple[int, int]) -> bool:
        """Check if the detected object is within the detection zone."""
        if not self.detection_zone['enabled']:
            return True
            
        height, width = frame_shape[:2]
        margin = self.detection_zone['margin']
        
        # Calculate zone boundaries
        zone_x1 = margin
        zone_y1 = margin
        zone_x2 = width - margin
        zone_y2 = height - margin
        
        # Check if the object is mostly within the zone
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (zone_x1 <= center_x <= zone_x2 and 
                zone_y1 <= center_y <= zone_y2)

    def _is_valid_bottle_shape(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if the detected object has a valid bottle shape."""
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size
        if width < self.min_detection_size or height < self.min_detection_size:
            print(f"Invalid size: {width}x{height} (min: {self.min_detection_size})")
            return False
            
        # Check aspect ratio (bottles are typically taller than they are wide)
        aspect_ratio = height / width
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            print(f"Invalid aspect ratio: {aspect_ratio:.2f} (range: {self.aspect_ratio_range})")
            return False
            
        # Check if the object is too large (likely not a bottle)
        max_size = 600  # Increased maximum size for a bottle
        if width > max_size or height > max_size:
            print(f"Object too large: {width}x{height} (max: {max_size})")
            return False
            
        return True

    def _is_stable_detection(self, track_id: int) -> bool:
        """Check if the detection is stable enough for cart addition."""
        if track_id not in self.confidence_history or track_id not in self.detection_history:
            return False
            
        # Check if we have enough history
        if len(self.confidence_history[track_id]) < self.min_confidence_history:
            return False
            
        # Check if confidence is consistently high enough
        recent_confidences = self.confidence_history[track_id][-self.min_confidence_history:]
        if not all(conf >= self.min_confidence for conf in recent_confidences):
            return False
            
        # Check if position is stable
        if len(self.detection_history[track_id]) >= 2:
            position_changes = [abs(self.detection_history[track_id][i] - self.detection_history[track_id][i-1]) 
                             for i in range(1, len(self.detection_history[track_id]))]
            avg_change = sum(position_changes) / len(position_changes)
            return avg_change < self.stability_threshold
            
        return False

    def detect_bottles(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect bottles in the frame and track their movements."""
        self.frame_count += 1
        self.debug_info = {}  # Reset debug info
        bottle_events = []

        # Preprocess frame
        original_frame = frame.copy()
        frame = cv2.resize(frame, (640, 640))  # Resize for better detection
        
        # Apply image enhancement
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=15)  # Increased contrast and brightness
        frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Reduced blur for sharper edges

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]

            # Filter detections based on confidence range
            valid_detections = []
            for i in range(len(results.boxes)):
                conf = results.boxes.conf[i].item()
                if self.conf_threshold <= conf <= self.max_conf_threshold:
                    valid_detections.append(i)

            # Store debug information
            self.debug_info['raw_detections'] = len(valid_detections)
            self.debug_info['confidences'] = [results.boxes.conf[i].item() for i in valid_detections]
            
            # Print detection info every 30 frames
            if self.frame_count % 30 == 0:
                print(f"\nDetection Info:")
                print(f"Raw detections: {len(valid_detections)}")
                if len(valid_detections) > 0:
                    print(f"Confidences: {[f'{results.boxes.conf[i].item():.2f}' for i in valid_detections]}")
                    print(f"Classes: {[self.model.names[int(results.boxes.cls[i])] for i in valid_detections]}")

            # Convert results to supervision format with filtered detections
            detections = sv.Detections(
                xyxy=results.boxes.xyxy[valid_detections].cpu().numpy(),
                confidence=results.boxes.conf[valid_detections].cpu().numpy(),
                class_id=results.boxes.cls[valid_detections].cpu().numpy().astype(int),
                tracker_id=results.boxes.id[valid_detections].cpu().numpy().astype(int) if results.boxes.id is not None else None
            )

            # Process detections
            annotated_frame = original_frame.copy()

            # Draw detection zone
            annotated_frame = self._draw_detection_zone(annotated_frame)

            # Mark all current objects as not in frame
            for track_id in self.tracked_objects:
                self.tracked_objects[track_id]['in_frame'] = False

            # Process new detections
            for i in range(len(detections)):
                box = detections.xyxy[i]
                conf = detections.confidence[i]
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else i
                class_id = detections.class_id[i]
                
                # Get bottle position
                x1, y1, x2, y2 = map(int, box)
                
                # Skip if not a valid bottle shape
                if not self._is_valid_bottle_shape(x1, y1, x2, y2):
                    continue
                    
                # Skip if not in detection zone
                if not self._is_in_detection_zone(x1, y1, x2, y2, frame.shape):
                    print(f"Bottle detected outside detection zone")
                    continue
                    
                center_y = (y1 + y2) / 2

                # Initialize tracking for new objects
                if track_id not in self.tracked_objects:
                    self.tracked_objects[track_id] = {
                        'product_id': None,
                        'last_seen': time.time(),
                        'position': center_y,
                        'in_frame': True,
                        'missing_frames': 0,
                        'in_cart': False,
                        'stable_frames': 0,
                        'class_id': class_id,
                        'removal_start_time': None  # Track when bottle started being out of view
                    }
                    self.last_positions[track_id] = center_y
                    self.stable_frames[track_id] = 0
                    self.detection_history[track_id] = []
                    self.confidence_history[track_id] = []

                # Update tracking
                self.tracked_objects[track_id]['last_seen'] = time.time()
                self.tracked_objects[track_id]['position'] = center_y
                self.tracked_objects[track_id]['in_frame'] = True
                self.tracked_objects[track_id]['missing_frames'] = 0
                self.tracked_objects[track_id]['removal_start_time'] = None  # Reset removal timer

                # Update detection history
                self.detection_history[track_id].append(center_y)
                if len(self.detection_history[track_id]) > self.max_history_size:
                    self.detection_history[track_id].pop(0)

                # Update confidence history
                self.confidence_history[track_id].append(conf)
                if len(self.confidence_history[track_id]) > self.max_history_size:
                    self.confidence_history[track_id].pop(0)

                # Add to cart if bottle is stable and not already in cart
                if (not self.tracked_objects[track_id]['in_cart'] and 
                    self._is_stable_detection(track_id)):
                    # Extract bottle image for verification
                    bottle_image = frame[y1:y2, x1:x2]
                    product_id = self._identify_product(bottle_image)
                    
                    if product_id:
                        self.tracked_objects[track_id]['product_id'] = product_id
                        self.tracked_objects[track_id]['in_cart'] = True
                        self.db.add_to_cart(product_id)
                        bottle_events.append({
                            'type': 'pickup',
                            'product_id': product_id,
                            'track_id': track_id,
                            'confidence': conf
                        })
                        print(f"\nBottle added to cart! Track ID: {track_id}, Confidence: {conf:.2f}")

                # Update last position
                self.last_positions[track_id] = center_y

                # Draw detection box and label
                color = (0, 255, 0) if self.tracked_objects[track_id]['in_cart'] else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Get product info if available
                product_info = ""
                if self.tracked_objects[track_id]['product_id']:
                    product = self.db.get_product(self.tracked_objects[track_id]['product_id'])
                    if product:
                        product_info = f"{product.get('product_name', 'Unknown')} - ₹{product.get('price', 0.0):.2f}"

                # Draw label with confidence and stability
                class_name = self.model.names[class_id]
                stability = "Stable" if self._is_stable_detection(track_id) else "Unstable"
                label = f"{class_name} - ID: {track_id} - {product_info} - Conf: {conf:.2f} - {stability}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Check for removed bottles
            current_time = time.time()
            for track_id, obj in list(self.tracked_objects.items()):
                if not obj['in_frame']:
                    # Start or update removal timer
                    if obj['removal_start_time'] is None:
                        obj['removal_start_time'] = current_time
                        print(f"\nBottle {track_id} out of view, starting removal timer...")
                    elif current_time - obj['removal_start_time'] >= self.removal_timeout:
                        # Remove from cart if it was in cart
                        if obj['in_cart'] and obj['product_id']:
                            product_id = obj['product_id']
                            self.db.remove_from_cart(product_id)
                            bottle_events.append({
                                'type': 'removed',
                                'product_id': product_id,
                                'track_id': track_id
                            })
                            print(f"\nBottle removed from cart! Track ID: {track_id}")
                        
                        # Clean up tracking data
                        del self.tracked_objects[track_id]
                        if track_id in self.last_positions:
                            del self.last_positions[track_id]
                        if track_id in self.stable_frames:
                            del self.stable_frames[track_id]
                        if track_id in self.detection_history:
                            del self.detection_history[track_id]
                        if track_id in self.confidence_history:
                            del self.confidence_history[track_id]
                else:
                    # Reset removal timer if bottle is back in frame
                    obj['removal_start_time'] = None

            # Add debug information to frame
            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if len(detections) > 0:
                cv2.putText(annotated_frame, f"Max Conf: {max(detections.confidence):.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return annotated_frame, bottle_events

        except Exception as e:
            print(f"Error in detect_bottles: {str(e)}")
            return original_frame, []

    def _identify_product(self, bottle_image: np.ndarray) -> Optional[str]:
        """Identify the specific product from the bottle image."""
        try:
            if self.uploaded_bottle_features is None:
                print("No uploaded bottle features available for comparison")
                return None
            
            # Resize image to fixed size
            bottle_image = cv2.resize(bottle_image, (224, 224))
            
            # Convert to grayscale
            gray = cv2.cvtColor(bottle_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply additional preprocessing
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Extract HOG features
            features = self.feature_extractor.compute(thresh)
            
            # Calculate similarity with uploaded bottle
            similarity = np.dot(features.flatten(), self.uploaded_bottle_features.flatten()) / (
                np.linalg.norm(features) * np.linalg.norm(self.uploaded_bottle_features)
            )
            
            print(f"\nBottle similarity: {similarity:.2f}")
            
            # Only return bottle ID if similarity is high enough
            if similarity >= self.similarity_threshold:
                return "BOTTLE_001"  # Return a default bottle ID
            else:
                print(f"Bottle not matched with uploaded image (similarity: {similarity:.2f})")
                return None
            
        except Exception as e:
            print(f"Error in bottle identification: {str(e)}")
            return None

    def _identify_product(self, headphone_image: np.ndarray) -> Optional[str]:
        """Identify if the detected object matches the reference headphone."""
        try:
            if not self.reference_features:
                print("No reference headphone features available for comparison")
                return None
            
            # Resize image to fixed size
            headphone_image = cv2.resize(headphone_image, (224, 224))
            
            # Convert to grayscale
            gray = cv2.cvtColor(headphone_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            equalized = cv2.equalizeHist(gray)
            
            # Apply additional preprocessing
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Extract HOG features
            features = self.feature_extractor.compute(thresh)
            
            # Calculate similarity with all reference images
            matches = 0
            for ref_features in self.reference_features:
                similarity = np.dot(features.flatten(), ref_features.flatten()) / (
                    np.linalg.norm(features) * np.linalg.norm(ref_features)
                )
                if similarity >= self.similarity_threshold:
                    matches += 1
            
            print(f"\nHeadphone matches: {matches}/{len(self.reference_features)}")
            
            # Only return headphone ID if enough reference images match
            if matches >= self.min_matches_required:
                return "HP_001"  # Return the headphone ID
            else:
                print(f"Object does not match reference headphones (matches: {matches})")
                return None
            
        except Exception as e:
            print(f"Error in headphone identification: {str(e)}")
    def get_active_cart(self) -> List[Dict]:
        """Get the current active cart items."""
        return self.db.get_active_cart()

    def get_cart_history(self, limit: int = 100) -> List[Dict]:
        """Get the cart history."""
        return self.db.get_cart_history(limit)

    def train_custom_model(self, dataset_path: str, epochs: int = 100):
        """Train a custom model on the Milton bottle dataset."""
        # Create dataset directory structure
        dataset_dir = Path(dataset_path)
        train_dir = dataset_dir / "images" / "train"
        val_dir = dataset_dir / "images" / "val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # Train the model
        self.model.train(
            data=str(dataset_dir / "dataset.yaml"),
            epochs=epochs,
            imgsz=640,
            batch=16,
            name="milton_bottle_model"
        )

        # Update model path to the trained model
        self.model = YOLO("runs/train/milton_bottle_model/weights/best.pt")
        self.model.to(self.device) 