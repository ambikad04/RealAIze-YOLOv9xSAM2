import cv2
import argparse
from bottle_detector import BottleDetector
from database import SmartRetailDB
import json
import os
import time

def test_single_image(image_path: str, model_path: str):
    """Test the bottle detector on a single image."""
    # Initialize database and load products
    db = SmartRetailDB()
    
    # Clear existing database
    if os.path.exists("smart_retail.db"):
        os.remove("smart_retail.db")
    
    # Initialize new database
    db.init_database()
    
    # Load products
    with open('products.json', 'r') as f:
        products = json.load(f)
        for product in products['products']:
            db.add_product(product)
            print(f"Added product: {product['product_name']}")
            time.sleep(0.1)  # Add small delay to prevent database locking

    # Initialize bottle detector
    detector = BottleDetector(
        model_path=model_path,
        conf_threshold=0.3,  # Lower threshold for testing
        iou_threshold=0.45
    )

    # Read the test image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    print("\nProcessing image...")
    print("Press any key to continue after each step")

    # Detect bottles
    annotated_frame, events = detector.detect_bottles(frame)

    # Display the results
    cv2.imshow('Bottle Detection', annotated_frame)
    cv2.waitKey(0)

    # Process events
    for event in events:
        if event['type'] == 'pickup':
            print(f"\nProduct picked up: {event['product_id']}")
            product = db.get_product(event['product_id'])
            if product:
                print(f"Product details:")
                print(f"- Name: {product['product_name']}")
                print(f"- Company: {product['company_name']}")
                print(f"- Price: ₹{product['price']}")
                print(f"- Features: {', '.join(product['metadata']['features'])}")
        elif event['type'] == 'putback':
            print(f"\nProduct put back: {event['product_id']}")

    # Show cart status
    cart = detector.get_active_cart()
    print("\nCurrent Cart:")
    for item in cart:
        print(f"- {item['product_name']} (₹{item['price']})")

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Test Bottle Detection')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model')
    args = parser.parse_args()

    test_single_image(args.image, args.model)

if __name__ == "__main__":
    main() 