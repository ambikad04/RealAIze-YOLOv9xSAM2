import cv2
import numpy as np
from database import SmartRetailDB
import time
from datetime import datetime
import os
from pathlib import Path
import torch

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S")

class MultiObjectDetector:
    def __init__(
        self,
        model_path: str,
        db_path: str = "smart_retail.db",
        conf_threshold: float = 0.35,
        max_conf_threshold: float = 0.75,
        iou_threshold: float = 0.45,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        print("\nInitializing Object Detector...")
        
        # Initialize database
        self.db = SmartRetailDB(db_path)
        
        # Load reference images for template matching
        self.templates = {
            'headphone': self._load_templates('headphones'),
            'bottle': self._load_templates('bottles')
        }
        
        # Product information
        self.products = {
            'headphone': {
                'id': 'HEADPHONE_001',
                'name': 'HP Gaming Headphone',
                'price': 1599.00
            },
            'bottle': {
                'id': 'BOTTLE_001',
                'name': 'Milton Bottle',
                'price': 499.00
            }
        }
        
        # Detection parameters - Made more lenient
        self.template_threshold = 0.4  # Lowered to 40% match
        self.min_confidence = 0.4  # Lowered to 40%
        self.scale_factors = [1.0]  # Single scale for speed
        self.min_template_matches = 1  # Reduced to 1
        
        # Motion detection variables
        self.last_frame = None
        self.motion_threshold = 20
        self.object_positions = {}
        
        # Performance optimization
        self.frame_skip = 2  # Process every other frame
        self.frame_count = 0
        self.last_detection_timestamp = time.time()
        self.detection_interval = 0.03  # 30ms between detections
        
        # Tracking variables
        self.objects_in_cart = {
            'headphone': False,
            'bottle': False
        }
        self.object_last_seen = {
            'headphone': time.time(),
            'bottle': time.time()
        }
        self.stable_detection_count = {
            'headphone': 0,
            'bottle': 0
        }
        
        # Minimum stable detections before adding to cart
        self.min_stable_detections = 3  # Reduced from 5 to 3
        
        print(f"\nLoaded templates:")
        print(f"Headphones: {len(self.templates['headphone'])} templates")
        print(f"Bottles: {len(self.templates['bottle'])} templates")

    def _load_templates(self, object_type):
        """Load reference images for template matching with proper preprocessing."""
        templates = []
        template_dir = Path(f"uploads/{object_type}")
        
        if not template_dir.exists():
            print(f"Warning: No templates found for {object_type}")
            return templates
            
        for img_path in template_dir.glob("*.jpg"):
            try:
                # Load and preprocess template
                template = cv2.imread(str(img_path))
                if template is not None:
                    # Convert to grayscale
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    
                    # Resize to a consistent size
                    template = cv2.resize(template, (200, 200))
                    
                    # Add template with its original path for debugging
                    templates.append({
                        'image': template,
                        'path': str(img_path)
                    })
                    print(f"Loaded template: {img_path.name}")
            except Exception as e:
                print(f"Error loading template {img_path}: {e}")
                
        return templates

    def _detect_motion(self, frame, x1, y1, x2, y2, track_id):
        """Detect if the object has moved significantly."""
        if self.last_frame is None:
            self.last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return True
            
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference between frames
        frame_diff = cv2.absdiff(self.last_frame, gray)
        
        # Get ROI (Region of Interest) around the detected object
        roi = frame_diff[y1:y2, x1:x2]
        
        # Calculate motion in ROI
        motion = np.mean(roi)
        
        # Update last frame
        self.last_frame = gray
        
        # Check if object has moved significantly
        if track_id not in self.object_positions:
            self.object_positions[track_id] = (x1, y1, x2, y2)
            return True
            
        # Calculate position change
        last_pos = self.object_positions[track_id]
        pos_change = abs(x1 - last_pos[0]) + abs(y1 - last_pos[1])
        
        # Update position
        self.object_positions[track_id] = (x1, y1, x2, y2)
        
        return motion > self.motion_threshold or pos_change > 10

    def _quick_validate_detection(self, frame, box, obj_type, template_path=None):
        """Simplified validation focusing on basic criteria."""
        x1, y1, x2, y2 = box
        
        # Check if detection is in the zone
        if not self._is_in_detection_zone(x1, y1, x2, y2, frame.shape):
            return False
        
        # Basic size check
        height = y2 - y1
        width = x2 - x1
        min_size = frame.shape[1] * 0.1  # 10% of frame width
        max_size = frame.shape[1] * 0.8  # 80% of frame width
        
        if width < min_size or height < min_size or width > max_size or height > max_size:
            return False
        
        # Basic aspect ratio check
        aspect_ratio = width / height
        if obj_type == 'headphone':
            # More lenient aspect ratio for headphones
            if not (1.0 <= aspect_ratio <= 4.0):
                return False
        elif obj_type == 'bottle':
            # More lenient aspect ratio for bottles
            if not (0.2 <= aspect_ratio <= 1.0):
                return False
        
        return True

    def detect_objects(self, frame):
        """Detect objects in the frame using simplified template matching."""
        # Skip frames for better performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return []
        
        # Check if enough time has passed since last detection
        current_time = time.time()
        if current_time - self.last_detection_timestamp < self.detection_interval:
            return []
        self.last_detection_timestamp = current_time
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Quick preprocessing
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Resize frame for faster processing
        scale_factor = 0.5
        small_gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor)
        
        detections = []
        detected_objects = {'headphone': False, 'bottle': False}
        
        # Detect each object type
        for obj_type, templates in self.templates.items():
            if not templates:
                continue
            
            # Try each template
            for template_data in templates:
                template = template_data['image']
                template_path = template_data['path']
                
                # Perform template matching
                result = cv2.matchTemplate(small_gray, template, cv2.TM_CCOEFF_NORMED)
                
                # Find locations where matching exceeds threshold
                locations = np.where(result >= self.template_threshold)
                
                for pt in zip(*locations[::-1]):
                    # Calculate confidence
                    confidence = result[pt[1], pt[0]]
                    
                    # Scale coordinates back to original size
                    x1 = int(pt[0] / scale_factor)
                    y1 = int(pt[1] / scale_factor)
                    x2 = int((pt[0] + template.shape[1]) / scale_factor)
                    y2 = int((pt[1] + template.shape[0]) / scale_factor)
                    
                    # Basic validation
                    if self._quick_validate_detection(frame, (x1, y1, x2, y2), obj_type, template_path):
                        detections.append({
                            'type': obj_type,
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2),
                            'template': template_path
                        })
                        detected_objects[obj_type] = True
                        self.stable_detection_count[obj_type] += 1
                        self.object_last_seen[obj_type] = time.time()
                        
                        # Add to cart if not already in cart and detection is stable
                        if not self.objects_in_cart[obj_type] and self.stable_detection_count[obj_type] >= self.min_stable_detections:
                            product = self.products[obj_type]
                            self.db.add_to_cart(product['id'])
                            self.objects_in_cart[obj_type] = True
                            print(f"\n[{get_timestamp()}] {product['name']} added to cart!")
                            print(f"Product: {product['name']}")
                            print(f"Price: ₹{product['price']:.2f}")
                            print(f"Matched template: {template_path}")
                            print(f"Confidence: {confidence:.2f}")
                            
                            # Calculate and display cart total
                            cart_items = self.db.get_active_cart()
                            total = sum(item['price'] for item in cart_items)
                            print(f"Cart Total: ₹{total:.2f}")
                else:
                    # Reset stable detection count if no detections found
                    self.stable_detection_count[obj_type] = 0
        
        # Sort detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove overlapping detections
        filtered_detections = []
        for det in detections:
            overlap = False
            for existing in filtered_detections:
                if det['type'] == existing['type']:
                    # Calculate overlap
                    x1 = max(det['box'][0], existing['box'][0])
                    y1 = max(det['box'][1], existing['box'][1])
                    x2 = min(det['box'][2], existing['box'][2])
                    y2 = min(det['box'][3], existing['box'][3])
                    
                    # Calculate intersection area
                    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
                    det_area = (det['box'][2] - det['box'][0]) * (det['box'][3] - det['box'][1])
                    existing_area = (existing['box'][2] - existing['box'][0]) * (existing['box'][3] - existing['box'][1])
                    
                    # Calculate overlap ratio
                    overlap_ratio = intersection_area / min(det_area, existing_area)
                    
                    if overlap_ratio > 0.3:  # 30% overlap threshold
                        overlap = True
                        # Keep the detection with higher confidence
                        if det['confidence'] > existing['confidence']:
                            filtered_detections.remove(existing)
                            filtered_detections.append(det)
                        break
            
            if not overlap:
                filtered_detections.append(det)
        
        return filtered_detections

    def _is_in_detection_zone(self, x1: int, y1: int, x2: int, y2: int, frame_shape) -> bool:
        """Check if the detected object is within the detection zone."""
        height, width = frame_shape[:2]
        
        # Define detection zone (center 60% of the frame)
        zone_margin = int(width * 0.2)  # Increased detection zone
        zone_left = zone_margin
        zone_right = width - zone_margin
        zone_top = int(height * 0.2)    # Increased detection zone
        zone_bottom = int(height * 0.8)  # Increased detection zone
        
        # Calculate object center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Check if object is within zone
        in_zone = (zone_left <= center_x <= zone_right and 
                  zone_top <= center_y <= zone_bottom)
        
        # Check object size (must be at least 5% of frame width)
        min_size = width * 0.05  # Reduced to 5%
        object_width = x2 - x1
        object_height = y2 - y1
        
        return in_zone and object_width >= min_size and object_height >= min_size

def list_available_cameras():
    """List all available cameras and their indices with detailed diagnostics."""
    available_cameras = []
    
    print("\n=== Camera Diagnostic Information ===")
    
    # 1. Check OpenCV version and build information
    print("\nOpenCV Information:")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Build Information: {cv2.getBuildInformation()}")
    
    # 2. Check system camera information
    print("\nSystem Camera Information:")
    try:
        import win32com.client
        wmi = win32com.client.GetObject("winmgmts:")
        cameras = wmi.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE (PNPClass = 'Camera' OR PNPClass = 'Image')")
        
        print("\nDetected USB cameras:")
        usb_cameras = []
        for camera in cameras:
            if camera.Name:
                camera_info = {
                    'name': camera.Name,
                    'device_id': camera.DeviceID,
                    'manufacturer': camera.Manufacturer if hasattr(camera, 'Manufacturer') else 'Unknown',
                    'status': camera.Status if hasattr(camera, 'Status') else 'Unknown'
                }
                usb_cameras.append(camera_info)
                print(f"\nCamera: {camera_info['name']}")
                print(f"   Manufacturer: {camera_info['manufacturer']}")
                print(f"   Device ID: {camera_info['device_id']}")
                print(f"   Status: {camera_info['status']}")
    except Exception as e:
        print(f"Warning: Could not get USB camera information: {e}")
        usb_cameras = []
    
    # 3. Test each camera index with different backends
    print("\nTesting camera indices...")
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for i in range(10):  # Try first 10 indices
        print(f"\nTesting camera index {i}:")
        for backend, backend_name in backends:
            try:
                print(f"  Trying {backend_name} backend...")
                cap = cv2.VideoCapture(i, backend)
                
                if not cap.isOpened():
                    print(f"  Failed to open with {backend_name}")
                    continue
                
                # Test reading a frame
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print(f"  Could not read frame with {backend_name}")
                    cap.release()
                    continue
                
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                format = cap.get(cv2.CAP_PROP_FOURCC)
                format_str = ''.join([chr((int(format) >> 8 * i) & 0xFF) for i in range(4)])
                
                print(f"  Successfully opened with {backend_name}")
                print(f"    Resolution: {width}x{height}")
                print(f"    FPS: {fps}")
                print(f"    Format: {format_str}")
                
                # Try to identify if this is a USB camera
                is_usb = False
                camera_name = f"Camera {i}"
                device_id = ""
                manufacturer = "Unknown"
                
                # Match with USB camera list
                for usb_cam in usb_cameras:
                    if str(i) in usb_cam['device_id']:
                        camera_name = usb_cam['name']
                        device_id = usb_cam['device_id']
                        manufacturer = usb_cam['manufacturer']
                        is_usb = True
                        break
                
                available_cameras.append({
                    'index': i,
                    'name': camera_name,
                    'is_usb': is_usb,
                    'device_id': device_id,
                    'manufacturer': manufacturer,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'backend': backend,
                    'backend_name': backend_name,
                    'format': format_str
                })
                
                cap.release()
                break
                
            except Exception as e:
                print(f"  Error with {backend_name}: {str(e)}")
                if cap is not None:
                    cap.release()
                continue
    
    # 4. Print summary
    print("\n=== Camera Summary ===")
    if available_cameras:
        print(f"\nFound {len(available_cameras)} working cameras:")
        for i, camera in enumerate(available_cameras):
            camera_type = "USB" if camera['is_usb'] else "Built-in"
            print(f"\n{i+1}. {camera['name']} ({camera_type})")
            print(f"   Index: {camera['index']}")
            print(f"   Resolution: {camera['resolution']} @ {camera['fps']} FPS")
            print(f"   Backend: {camera['backend_name']}")
            print(f"   Format: {camera['format']}")
            if camera['device_id']:
                print(f"   Device ID: {camera['device_id']}")
            if camera['manufacturer'] != "Unknown":
                print(f"   Manufacturer: {camera['manufacturer']}")
    else:
        print("\nNo working cameras found!")
    
    return available_cameras

def main():
    # Initialize detector and database
    model_path = "yolov8n.pt"
    detector = MultiObjectDetector(model_path=model_path)
    
    print("\n=== Multi-Object Detection System ===")
    
    # List available cameras first
    print("\nScanning for available cameras...")
    available_cameras = list_available_cameras()
    
    if not available_cameras:
        print("\nError: No cameras found!")
        return
    
    # Try to find USB camera
    usb_camera = None
    for camera in available_cameras:
        if camera['is_usb']:
            usb_camera = camera
            break
    
    # Initialize camera with multiple attempts
    cap = None
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            print(f"\nAttempt {attempt + 1} to initialize camera...")
            
            # Try USB camera first if available
            if usb_camera:
                print(f"Trying USB camera: {usb_camera['name']}")
                cap = cv2.VideoCapture(usb_camera['index'], cv2.CAP_DSHOW)
            else:
                print("No USB camera found, trying default camera...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                print("Failed with DirectShow, trying default backend...")
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise Exception("Could not open camera")
            
            # Test camera with a frame read
            ret, test_frame = cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                raise Exception("Could not read frame from camera")
            
            print("Camera initialized successfully!")
            break
            
        except Exception as e:
            print(f"Error during camera initialization: {str(e)}")
            if cap is not None:
                cap.release()
            attempt += 1
            if attempt < max_attempts:
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("\nFailed to initialize camera after multiple attempts!")
                return
    
    # Set camera properties with error checking
    print("\nConfiguring camera settings...")
    try:
        # Set basic properties first
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test if settings were applied
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera settings applied:")
        print(f"Resolution: {actual_width}x{actual_height}")
        print(f"FPS: {actual_fps}")
        
        # Try to set additional properties
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
    except Exception as e:
        print(f"Warning: Some camera settings could not be applied: {str(e)}")
        print("Continuing with default settings...")
    
    # Create a resizable window
    cv2.namedWindow('Multi-Object Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Multi-Object Detection', 640, 480)
    
    print("\n=== System Ready ===")
    print("Press 'q' to quit")
    print("Press 'r' to reset detection")
    print("Press 's' to save frame")
    print("Press 't' to test detection")
    print("Press 'c' to reconnect camera")
    
    frame_count = 0
    last_time = time.time()
    consecutive_failures = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                print(f"Error: Could not read frame (attempt {consecutive_failures})")
                if consecutive_failures >= 5:
                    print("Too many consecutive failures. Attempting to reconnect camera...")
                    cap.release()
                    time.sleep(1)  # Wait a bit before reconnecting
                    
                    # Try to reconnect
                    if usb_camera:
                        cap = cv2.VideoCapture(usb_camera['index'], cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    
                    if not cap.isOpened():
                        print("Failed to reconnect camera. Please check connection and restart the program.")
                        break
                    
                    consecutive_failures = 0
                    print("Camera reconnected successfully!")
                continue
            
            consecutive_failures = 0  # Reset failure counter on successful frame
            
            # Create a copy for drawing
            display_frame = frame.copy()
            
            # Draw detection zone
            height, width = frame.shape[:2]
            margin = int(width * 0.2)  # 20% margin
            cv2.rectangle(display_frame, (margin, margin), 
                         (width - margin, height - margin), 
                         (0, 255, 255), 2)
            
            # Detect objects
            detections = detector.detect_objects(frame)
            
            # Process detections
            detected_objects = {'headphone': False, 'bottle': False}
            
            for det in detections:
                obj_type = det['type']
                confidence = det['confidence']
                x1, y1, x2, y2 = det['box']
                
                # Draw bounding box with confidence
                color = (0, 255, 0)  # Green for detected objects
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add label with confidence
                label = f"{obj_type.title()} {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw object center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                detected_objects[obj_type] = True
            
            # Remove objects from cart if not detected for 0.3 seconds
            for obj_type in ['headphone', 'bottle']:
                if detector.objects_in_cart[obj_type] and time.time() - detector.object_last_seen[obj_type] > 0.3:
                    product = detector.products[obj_type]
                    detector.db.remove_from_cart(product['id'])
                    detector.objects_in_cart[obj_type] = False
                    detector.stable_detection_count[obj_type] = 0
                    print(f"\n[{get_timestamp()}] {product['name']} removed from cart!")
                    print(f"Product: {product['name']}")
                    print(f"Price: ₹{product['price']:.2f}")
                    
                    # Calculate and display cart total
                    cart_items = detector.db.get_active_cart()
                    total = sum(item['price'] for item in cart_items)
                    print(f"Cart Total: ₹{total:.2f}")
            
            # Display cart total and detection info
            cart_items = detector.db.get_active_cart()
            total = sum(item['price'] for item in cart_items)
            
            # Add status information
            cv2.putText(display_frame, f"Cart Total: ₹{total:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add instructions
            if not any(detected_objects.values()):
                cv2.putText(display_frame, "Place object in yellow zone", (10, height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                detected_text = "Detected: " + ", ".join(
                    obj.title() for obj, detected in detected_objects.items() if detected
                )
                cv2.putText(display_frame, detected_text, (10, height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Multi-Object Detection', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                print("\n=== Multi-Object Detection System Stopped ===")
                break
            elif key == ord('r'):
                # Reset tracking
                for obj_type in ['headphone', 'bottle']:
                    detector.objects_in_cart[obj_type] = False
                    detector.object_last_seen[obj_type] = time.time()
                    detector.stable_detection_count[obj_type] = 0
                print(f"\n[{get_timestamp()}] Detection system reset")
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'debug_frame_{timestamp}.jpg', frame)
                print(f"\nSaved debug frame as 'debug_frame_{timestamp}.jpg'")
            elif key == ord('t'):
                # Test detection
                print("\nTesting detection with current frame...")
                detections = detector.detect_objects(frame)
                print(f"Detection results: {len(detections)} objects detected")
                if detections:
                    for det in detections:
                        print(f"Detected {det['type']} with confidence {det['confidence']:.2f}")
                else:
                    print("No objects detected in current frame")
            elif key == ord('c'):
                # Manual camera reconnect
                print("\nAttempting to reconnect camera...")
                cap.release()
                time.sleep(1)
                
                if usb_camera:
                    cap = cv2.VideoCapture(usb_camera['index'], cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                
                if not cap.isOpened():
                    print("Failed to reconnect camera!")
                else:
                    print("Camera reconnected successfully!")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
    finally:
        # Cleanup
        print("\nCleaning up...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Additional wait to ensure windows are closed

if __name__ == "__main__":
    main() 