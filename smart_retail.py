import cv2
import argparse
import time
from bottle_detector import BottleDetector
from database import SmartRetailDB
import json
import os

def load_products(db: SmartRetailDB, products_file: str):
    """Load products from a JSON file into the database."""
    if not os.path.exists(products_file):
        print(f"Products file {products_file} not found")
        return

    with open(products_file, 'r') as f:
        products = json.load(f)
        for product in products:
            db.add_product(product)
            print(f"Added product: {product['product_name']}")

def main():
    parser = argparse.ArgumentParser(description='Smart Retail System')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    parser.add_argument('--products', type=str, default='products.json', help='Path to products JSON file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    args = parser.parse_args()

    # Initialize database and load products
    db = SmartRetailDB()
    load_products(db, args.products)

    # Initialize bottle detector
    detector = BottleDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return

    print("Smart Retail System Started")
    print("Press 'q' to quit")
    print("Press 'c' to show cart")
    print("Press 'h' to show history")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect bottles and track movements
        annotated_frame, events = detector.detect_bottles(frame)

        # Process events
        for event in events:
            if event['type'] == 'pickup':
                print(f"\nProduct picked up: {event['product_id']}")
            elif event['type'] == 'putback':
                print(f"\nProduct put back: {event['product_id']}")

        # Display the frame
        cv2.imshow('Smart Retail System', annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cart = detector.get_active_cart()
            print("\nCurrent Cart:")
            for item in cart:
                print(f"- {item['product_name']} (₹{item['price']})")
        elif key == ord('h'):
            history = detector.get_cart_history(10)
            print("\nRecent History:")
            for item in history:
                action = "Added" if item['action'] == 'add' else "Removed"
                print(f"- {action}: {item['product_name']} (₹{item['price']}) at {item['timestamp']}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 