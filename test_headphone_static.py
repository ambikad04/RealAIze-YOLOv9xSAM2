import cv2
from ultralytics import YOLO
import os

def main():
    # Initialize the model
    model_path = 'runs/detect/headphone_model2/weights/best.pt'
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # List all images in the uploads/headphones directory
    image_dir = 'uploads/headphones'
    if not os.path.exists(image_dir):
        print(f"Error: Directory {image_dir} does not exist")
        return
        
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return
        
    print(f"\nFound {len(image_files)} images:")
    for img_file in image_files:
        print(f"- {img_file}")
    
    # Test each image
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\nTesting image: {img_file}")
        
        # Load and display image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image {img_path}")
            continue
            
        # Run detection with very low confidence threshold
        results = model(img, conf=0.05, verbose=False)[0]
        
        # Print results
        print(f"Number of detections: {len(results.boxes)}")
        if len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                print(f"Detection {i+1}:")
                print(f"  Confidence: {conf:.3f}")
                print(f"  Box: ({x1}, {y1}, {x2}, {y2})")
                print(f"  Size: {x2-x1}x{y2-y1}")
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"Headphone {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No detections found!")
        
        # Show image
        cv2.imshow(f'Test - {img_file}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 