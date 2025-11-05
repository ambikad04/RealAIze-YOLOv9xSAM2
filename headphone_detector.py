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

class HeadphoneDetector:
    def __init__(
        self,
        model_path: str,
        db_path: str = "smart_retail.db",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print(f"Loading model from: {model_path}")
        print(f"Using device: {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.db = SmartRetailDB(db_path)
        self.device = device
        
        # Print model information
        print(f"Model loaded successfully")
        print(f"Model classes: {self.model.names}")
        print(f"Confidence threshold: {self.conf_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        
        # Tracking variables
        self.tracked_objects = {}
        self.pickup_threshold = 0.35
        self.putback_threshold = 0.35
        self.last_positions = {}
        self.frame_count = 0
        self.missing_frames_threshold = 2
        self.stable_frames_threshold = 4
        self.stable_frames = {}
        self.detection_history = {}
        self.max_history_size = 5
        self.confidence_history = {}
        self.debug_info = {}
        self.uploaded_headphone_features = None
        self.similarity_threshold = 0.55
        self.last_cart_state = set()
        self.removal_timeout = 1.0
        self.min_detection_size = 60  # Adjusted for headphones
        self.aspect_ratio_range = (0.5, 2.0)  # Adjusted for headphones
        
        # Detection zone settings
        self.detection_zone = {
            'enabled': True,
            'margin': 100,
            'color': (0, 255, 255),
            'thickness': 2,
            'alpha': 0.3
        }
        
        # Stability tracking
        self.stability_threshold = 20
        self.min_confidence_history = 3
        self.min_confidence = 0.35
        
        # Feature extraction settings
        self.feature_extractor = cv2.HOGDescriptor(
            (224, 224),
            (16, 16),
            (8, 8),
            (8, 8),
            9
        )
        
        # Load uploaded headphone features if available
        self._load_uploaded_headphone_features()

    def _load_uploaded_headphone_features(self):
        """Load features from the uploaded headphone image."""
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            print("No uploads directory found")
            return
            
        # Find the most recent image in uploads
        image_files = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
        if not image_files:
            print("No headphone images found in uploads directory")
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
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Extract HOG features
            features = self.feature_extractor.compute(thresh)
            
            # Store features
            self.uploaded_headphone_features = features
            print("Successfully loaded uploaded headphone features")
            
        except Exception as e:
            print(f"Error loading uploaded headphone features: {str(e)}")

    def _draw_detection_zone(self, frame: np.ndarray) -> np.ndarray:
        """Draw the detection zone on the frame."""
        height, width = frame.shape[:2]
        margin = self.detection_zone['margin']
        
        x1 = margin
        y1 = margin
        x2 = width - margin
        y2 = height - margin
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.detection_zone['color'], -1)
        cv2.addWeighted(overlay, self.detection_zone['alpha'], frame, 1 - self.detection_zone['alpha'], 0, frame)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.detection_zone['color'], self.detection_zone['thickness'])
        
        cv2.putText(frame, "Place Headphone Here", (x1 + 10, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.detection_zone['color'], 2)
        
        return frame

    def _is_in_detection_zone(self, x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple[int, int]) -> bool:
        """Check if the detected object is within the detection zone."""
        if not self.detection_zone['enabled']:
            return True
            
        height, width = frame_shape[:2]
        margin = self.detection_zone['margin']
        
        zone_x1 = margin
        zone_y1 = margin
        zone_x2 = width - margin
        zone_y2 = height - margin
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return (zone_x1 <= center_x <= zone_x2 and 
                zone_y1 <= center_y <= zone_y2)

    def _is_valid_headphone_shape(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if the detected object has a valid headphone shape."""
        width = x2 - x1
        height = y2 - y1
        
        if width < self.min_detection_size or height < self.min_detection_size:
            print(f"Invalid size: {width}x{height} (min: {self.min_detection_size})")
            return False
            
        aspect_ratio = height / width
        if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
            print(f"Invalid aspect ratio: {aspect_ratio:.2f} (range: {self.aspect_ratio_range})")
            return False
            
        max_size = 500  # Adjusted for headphones
        if width > max_size or height > max_size:
            print(f"Object too large: {width}x{height} (max: {max_size})")
            return False
            
        return True

    def _is_stable_detection(self, track_id: int) -> bool:
        """Check if the detection is stable enough for cart addition."""
        if track_id not in self.confidence_history or track_id not in self.detection_history:
            return False
            
        if len(self.confidence_history[track_id]) < self.min_confidence_history:
            return False
            
        recent_confidences = self.confidence_history[track_id][-self.min_confidence_history:]
        if not all(conf >= self.min_confidence for conf in recent_confidences):
            return False
            
        if len(self.detection_history[track_id]) >= 2:
            position_changes = [abs(self.detection_history[track_id][i] - self.detection_history[track_id][i-1]) 
                             for i in range(1, len(self.detection_history[track_id]))]
            avg_change = sum(position_changes) / len(position_changes)
            return avg_change < self.stability_threshold
            
        return False

    def detect_headphones(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Detect headphones in the frame and track their movements."""
        self.frame_count += 1
        self.debug_info = {}
        headphone_events = []

        original_frame = frame.copy()
        frame = cv2.resize(frame, (640, 640))
        
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=15)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]

            self.debug_info['raw_detections'] = len(results.boxes)
            self.debug_info['confidences'] = results.boxes.conf.cpu().numpy().tolist() if len(results.boxes) > 0 else []
            
            if self.frame_count % 30 == 0:
                print(f"\nDetection Info:")
                print(f"Raw detections: {len(results.boxes)}")
                if len(results.boxes) > 0:
                    print(f"Confidences: {[f'{conf:.2f}' for conf in results.boxes.conf.cpu().numpy()]}")
                    print(f"Classes: {[self.model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]}")

            detections = sv.Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int),
                tracker_id=results.boxes.id.cpu().numpy().astype(int) if results.boxes.id is not None else None
            )

            annotated_frame = original_frame.copy()
            annotated_frame = self._draw_detection_zone(annotated_frame)

            for track_id in self.tracked_objects:
                self.tracked_objects[track_id]['in_frame'] = False

            for i in range(len(detections)):
                box = detections.xyxy[i]
                conf = detections.confidence[i]
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else i
                class_id = detections.class_id[i]
                
                x1, y1, x2, y2 = map(int, box)
                
                if not self._is_valid_headphone_shape(x1, y1, x2, y2):
                    continue
                    
                if not self._is_in_detection_zone(x1, y1, x2, y2, frame.shape):
                    print(f"Headphone detected outside detection zone")
                    continue
                    
                center_y = (y1 + y2) / 2

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
                        'removal_start_time': None
                    }
                    self.last_positions[track_id] = center_y
                    self.stable_frames[track_id] = 0
                    self.detection_history[track_id] = []
                    self.confidence_history[track_id] = []

                self.tracked_objects[track_id]['last_seen'] = time.time()
                self.tracked_objects[track_id]['position'] = center_y
                self.tracked_objects[track_id]['in_frame'] = True
                self.tracked_objects[track_id]['missing_frames'] = 0
                self.tracked_objects[track_id]['removal_start_time'] = None

                self.detection_history[track_id].append(center_y)
                if len(self.detection_history[track_id]) > self.max_history_size:
                    self.detection_history[track_id].pop(0)

                self.confidence_history[track_id].append(conf)
                if len(self.confidence_history[track_id]) > self.max_history_size:
                    self.confidence_history[track_id].pop(0)

                if (not self.tracked_objects[track_id]['in_cart'] and 
                    self._is_stable_detection(track_id)):
                    headphone_image = frame[y1:y2, x1:x2]
                    product_id = self._identify_product(headphone_image)
                    
                    if product_id:
                        self.tracked_objects[track_id]['product_id'] = product_id
                        self.tracked_objects[track_id]['in_cart'] = True
                        self.db.add_to_cart(product_id)
                        headphone_events.append({
                            'type': 'pickup',
                            'product_id': product_id,
                            'track_id': track_id,
                            'confidence': conf
                        })
                        print(f"\nHeadphone added to cart! Track ID: {track_id}, Confidence: {conf:.2f}")

                self.last_positions[track_id] = center_y

                color = (0, 255, 0) if self.tracked_objects[track_id]['in_cart'] else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                product_info = ""
                if self.tracked_objects[track_id]['product_id']:
                    product = self.db.get_product(self.tracked_objects[track_id]['product_id'])
                    if product:
                        product_info = f"{product.get('product_name', 'Unknown')} - ₹{product.get('price', 0.0):.2f}"

                class_name = self.model.names[class_id]
                stability = "Stable" if self._is_stable_detection(track_id) else "Unstable"
                label = f"{class_name} - ID: {track_id} - {product_info} - Conf: {conf:.2f} - {stability}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            current_time = time.time()
            for track_id, obj in list(self.tracked_objects.items()):
                if not obj['in_frame']:
                    if obj['removal_start_time'] is None:
                        obj['removal_start_time'] = current_time
                        print(f"\nHeadphone {track_id} out of view, starting removal timer...")
                    elif current_time - obj['removal_start_time'] >= self.removal_timeout:
                        if obj['in_cart'] and obj['product_id']:
                            product_id = obj['product_id']
                            self.db.remove_from_cart(product_id)
                            headphone_events.append({
                                'type': 'removed',
                                'product_id': product_id,
                                'track_id': track_id
                            })
                            print(f"\nHeadphone removed from cart! Track ID: {track_id}")
                        
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
                    obj['removal_start_time'] = None

            cv2.putText(annotated_frame, f"Detections: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if len(detections) > 0:
                cv2.putText(annotated_frame, f"Max Conf: {max(detections.confidence):.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return annotated_frame, headphone_events

        except Exception as e:
            print(f"Error in detect_headphones: {str(e)}")
            return original_frame, []

    def _identify_product(self, headphone_image: np.ndarray) -> Optional[str]:
        """Identify the specific product from the headphone image."""
        try:
            if self.uploaded_headphone_features is None:
                print("No uploaded headphone features available for comparison")
                return None
            
            headphone_image = cv2.resize(headphone_image, (224, 224))
            gray = cv2.cvtColor(headphone_image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            features = self.feature_extractor.compute(thresh)
            
            similarity = np.dot(features.flatten(), self.uploaded_headphone_features.flatten()) / (
                np.linalg.norm(features) * np.linalg.norm(self.uploaded_headphone_features)
            )
            
            print(f"\nHeadphone similarity: {similarity:.2f}")
            
            if similarity >= self.similarity_threshold:
                return "HEADPHONE_001"  # Return a default headphone ID
            else:
                print(f"Headphone not matched with uploaded image (similarity: {similarity:.2f})")
                return None
            
        except Exception as e:
            print(f"Error in headphone identification: {str(e)}")
            return None

    def get_active_cart(self) -> List[Dict]:
        """Get the current active cart items."""
        return self.db.get_active_cart()

    def get_cart_history(self, limit: int = 100) -> List[Dict]:
        """Get the cart history."""
        return self.db.get_cart_history(limit)

    def train_custom_model(self, dataset_path: str, epochs: int = 100):
        """Train a custom model on the headphone dataset."""
        dataset_dir = Path(dataset_path)
        train_dir = dataset_dir / "images" / "train"
        val_dir = dataset_dir / "images" / "val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        self.model.train(
            data=str(dataset_dir / "dataset.yaml"),
            epochs=epochs,
            imgsz=640,
            batch=16,
            name="headphone_model"
        )

        self.model = YOLO("runs/train/headphone_model/weights/best.pt")
        self.model.to(self.device) 