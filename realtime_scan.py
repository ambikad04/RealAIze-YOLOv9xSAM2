import cv2
import numpy as np
from bottle_detector import BottleDetector
from database import SmartRetailDB
import time
from datetime import datetime
from pathlib import Path

def format_price(price):
    """Format price with proper currency symbol and decimal places."""
    return f"₹{price:.2f}"

def print_product_details(product, action="added"):
    """Print detailed product information."""
    print("\n" + "="*50)
    print(f"Product {action.upper()}:")
    print("="*50)
    try:
        print(f"Name: {product.get('product_name', 'Unknown')}")
        print(f"Company: {product.get('company_name', 'Unknown')}")
        print(f"Category: {product.get('category', 'Bottles')}")
        print(f"Price: {format_price(product.get('price', 0.0))}")
        print(f"Description: {product.get('description', 'No description available')}")
    except Exception as e:
        print(f"Error displaying product details: {e}")
    print("="*50)

def main():
    # Find the trained model
    model_paths = [
        Path("models/best.pt"),  # First try backup location
        Path("models/last.pt"),  # Then try last model in backup
        Path("runs/detect/milton_bottle_model/weights/best.pt"),  # Then try original location
        Path("runs/detect/milton_bottle_model/weights/last.pt")   # Finally try last model in original location
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            print(f"Found model at: {path}")
            break
    
    if model_path is None:
        print("Error: No trained model found. Please run train_model.py first.")
        return

    print(f"Using model: {model_path}")

    # Initialize the bottle detector with more sensitive parameters
    detector = BottleDetector(
        model_path=str(model_path),
        conf_threshold=0.05,  # Very low confidence threshold
        iou_threshold=0.2     # Lower IoU threshold
    )

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.0)
    cap.set(cv2.CAP_PROP_CONTRAST, 1.0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Adjust exposure for better detection
    cap.set(cv2.CAP_PROP_GAIN, 1.0)     # Increase gain for better detection

    print("\nInstructions:")
    print("1. Press 'q' to quit the application")
    print("2. Show the Milton bottle to the camera")
    print("3. Hold the bottle steady for 1 second to add to cart")
    print("4. Remove from view to remove from cart")
    print("5. Multiple bottles will be tracked separately")
    print("\nDebug mode enabled - showing detection information")

    # Variables for tracking
    last_cart_update = time.time()
    cart_update_interval = 1.0  # Update cart status every second
    last_frame_time = time.time()
    fps = 0
    frame_count = 0
    tracked_bottles = {}  # Dictionary to track individual bottles
    bottle_counter = 0    # Counter for unique bottle IDs
    detection_start_time = {}  # Track when each bottle was first detected
    detection_threshold = 0.8  # Slightly reduced hold time for better responsiveness
    last_detection_time = {}   # Track last detection time for each bottle
    detection_timeout = 1.5    # Reduced timeout for better responsiveness
    debug_mode = True         # Enable debug mode
    consecutive_detections = {}  # Track consecutive detections for stability
    min_consecutive_detections = 2  # Reduced for faster detection
    detection_history = {}  # Track detection history for each bottle
    max_history_size = 8    # Balanced history size
    detection_confidence = {}  # Track confidence scores
    last_frame = None  # Store last frame for motion detection
    motion_threshold = 30  # Threshold for motion detection
    stable_detection_count = {}  # Count stable detections
    min_stable_detections = 2  # Minimum stable detections required
    no_detection_frames = 0  # Count frames with no detection
    max_no_detection_frames = 30  # Maximum frames without detection before warning
    last_debug_print = time.time()  # Track last debug print time
    debug_print_interval = 1.0  # Print debug info every second
    motion_percentage = 0.0  # Initialize motion percentage
    cart_count = 0  # Initialize cart count
    removal_timeout = 1.0  # Time to wait before removing bottle from cart

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Calculate FPS
            current_time = time.time()
            frame_count += 1
            if current_time - last_frame_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_frame_time = current_time

            # Detect bottles
            annotated_frame, events = detector.detect_bottles(frame)

            # Process events
            for event in events:
                if event['type'] == 'pickup':
                    product_id = event['product_id']
                    product = detector.db.get_product(product_id)
                    if product:
                        print_product_details(product, "added")
                        print(f"Detection confidence: {event.get('confidence', 0.0):.2f}")
                elif event['type'] == 'removed':
                    product_id = event['product_id']
                    product = detector.db.get_product(product_id)
                    if product:
                        print_product_details(product, "removed")

            # Update cart count
            active_cart = detector.get_active_cart()
            cart_count = len(active_cart)

            # Update no detection counter
            if len(events) == 0:
                no_detection_frames += 1
                if no_detection_frames >= max_no_detection_frames:
                    print("\nWarning: No bottle detected for a while. Try adjusting the bottle position or lighting.")
                    no_detection_frames = 0
            else:
                no_detection_frames = 0

            # Motion detection
            if last_frame is not None:
                # Convert frames to grayscale
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_last = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate absolute difference
                frame_diff = cv2.absdiff(gray_current, gray_last)
                
                # Apply threshold
                _, thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)
                
                # Calculate motion percentage
                motion_percentage = (np.sum(thresh) / 255) / thresh.size * 100

            # Update last frame
            last_frame = frame.copy()

            # Add FPS and instructions to the frame
            cv2.putText(annotated_frame, f"FPS: {fps}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Motion: {motion_percentage:.1f}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"No Detection: {no_detection_frames}/{max_no_detection_frames}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Cart Count: {cart_count}", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Add cart status to frame
            y_offset = 70
            cv2.putText(annotated_frame, "Cart Status:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if active_cart:
                total_price = 0
                for item in active_cart:
                    y_offset += 30
                    product = detector.db.get_product(item['product_id'])
                    if product:
                        total_price += product.get('price', 0.0)
                        cv2.putText(annotated_frame, 
                                   f"- {product.get('product_name', 'Unknown')} ({format_price(product.get('price', 0.0))})",
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 20
                        cv2.putText(annotated_frame, 
                                   f"  {product.get('company_name', 'Unknown')}",
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add total price
                y_offset += 30
                cv2.putText(annotated_frame, f"Total: {format_price(total_price)}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                y_offset += 30
                cv2.putText(annotated_frame, "Cart is empty", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Add detection status
            if detection_start_time:
                y_offset += 30
                cv2.putText(annotated_frame, "Hold bottle steady...", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Add debug information
            if debug_mode:
                y_offset += 30
                cv2.putText(annotated_frame, f"Debug: Events={len(events)}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Add quit instruction
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, annotated_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Print debug info periodically
            if current_time - last_debug_print >= debug_print_interval:
                print(f"\rMotion: {motion_percentage:.1f}% | FPS: {fps} | Events: {len(events)} | Cart: {cart_count}", end="")
                last_debug_print = current_time

            # Display the frame
            cv2.imshow('Milton Bottle Detector', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nQuitting application...")
                break

        except Exception as e:
            print(f"\nError in main loop: {str(e)}")
            continue

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main() 