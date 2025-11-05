import cv2
import numpy as np
from bottle_detector import BottleDetector
import time
from pathlib import Path
import os
from download_weights import download_weights
from database import SmartRetailDB

def find_android_camera():
    """Find the Android camera connected via USB."""
    # Try different camera indices
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Read a test frame
            ret, frame = cap.read()
            if ret:
                print(f"Found Android camera at index {i}")
                return i
            cap.release()
    return None

def start_mobile_scanner():
    """Start the mobile phone camera scanner."""
    print("Starting mobile phone camera scanner...")
    
    # Check if model exists, if not download it
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print("Model not found. Downloading...")
        if not download_weights():
            print("Failed to download model. Please check your internet connection.")
            return
        print("Model downloaded successfully!")
    
    # Initialize the bottle detector
    try:
        detector = BottleDetector(
            model_path=str(model_path),
            conf_threshold=0.25,
            iou_threshold=0.45
        )
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return
    
    # Try to connect to the camera
    cap = None
    
    # First try USB connection
    print("\nTrying to connect via USB...")
    camera_index = find_android_camera()
    if camera_index is not None:
        cap = cv2.VideoCapture(camera_index)
        print("Connected to Android camera via USB!")
    else:
        # If USB fails, try IP camera
        print("\nUSB connection failed. Trying IP camera...")
        print("Make sure you have the IP Webcam app installed and running on your phone.")
        camera_url = "http://192.168.1.100:8080/video"  # Default IP camera URL
        cap = cv2.VideoCapture(camera_url)
        
        if not cap.isOpened():
            print("\nError: Could not connect to mobile camera")
            print("\nConnection Options:")
            print("1. USB Connection:")
            print("   - Connect your phone via USB")
            print("   - Enable USB debugging on your phone")
            print("   - Allow USB debugging when prompted")
            print("\n2. IP Camera Connection:")
            print("   - Install 'IP Webcam' app on your phone")
            print("   - Connect your phone to the same WiFi as your computer")
            print("   - Open the app and start the server")
            print("   - Note the IP address shown in the app")
            print("   - Update the camera_url in this file with your phone's IP")
            return
    
    print("\nCamera connected successfully!")
    print("Press 'q' to quit")
    print("Place the bottle in the yellow box to scan")
    
    # Variables for FPS calculation
    prev_time = time.time()
    fps = 0
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
                
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # Detect bottles
            annotated_frame, bottle_events = detector.detect_bottles(frame)
            
            # Add FPS counter
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Handle bottle events
            for event in bottle_events:
                if event['type'] == 'pickup':
                    product = detector.db.get_product(event['product_id'])
                    if product:
                        print("\n" + "="*50)
                        print("Product ADDED to Cart:")
                        print("="*50)
                        print(f"Name: {product.get('product_name', 'Unknown')}")
                        print(f"Company: {product.get('company', 'Unknown')}")
                        print(f"Category: {product.get('category', 'Unknown')}")
                        print(f"Price: ₹{product.get('price', 0.0):.2f}")
                        print(f"Description: {product.get('description', 'No description')}")
                        print("="*50)
                        print(f"Detection confidence: {event.get('confidence', 0.0):.2f}")
                        print("\nCurrent Cart Status:")
                        print(f"Total Items: {len(detector.get_active_cart())}")
                        print(f"Total Value: ₹{sum(item.get('price', 0.0) for item in detector.get_active_cart()):.2f}")
                        print("="*50 + "\n")
                elif event['type'] == 'removed':
                    product = detector.db.get_product(event['product_id'])
                    if product:
                        print("\n" + "="*50)
                        print("Product REMOVED from Cart:")
                        print("="*50)
                        print(f"Name: {product.get('product_name', 'Unknown')}")
                        print(f"Company: {product.get('company', 'Unknown')}")
                        print(f"Category: {product.get('category', 'Unknown')}")
                        print(f"Price: ₹{product.get('price', 0.0):.2f}")
                        print(f"Description: {product.get('description', 'No description')}")
                        print("="*50)
                        print("\nUpdated Cart Status:")
                        print(f"Total Items: {len(detector.get_active_cart())}")
                        print(f"Total Value: ₹{sum(item.get('price', 0.0) for item in detector.get_active_cart()):.2f}")
                        print("="*50 + "\n")
            
            # Display cart status on frame
            cart_items = detector.get_active_cart()
            total_value = sum(item.get('price', 0.0) for item in cart_items)
            
            # Draw cart status
            cv2.putText(annotated_frame, f"Cart Items: {len(cart_items)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Total: ₹{total_value:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw instructions
            cv2.putText(annotated_frame, "Place bottle in yellow box to scan", (10, annotated_frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Remove bottle to update cart", (10, annotated_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow("Mobile Bottle Scanner", annotated_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error during scanning: {str(e)}")
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\nMobile scanner closed")

if __name__ == "__main__":
    start_mobile_scanner() 


    