import cv2
import torch
import argparse
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, List, Dict
import time
import uuid
import os
import json
import faiss
from huggingface_hub import login, HfFolder

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv

import config
from utils import (
    preprocess_image,
    postprocess_detections,
    draw_detections,
    get_device
)
from embedding_utils import EmbeddingManager

def set_hf_token(token: str):
    """Set Hugging Face token for authentication."""
    try:
        login(token=token, add_to_git_credential=False)
        HfFolder.save_token(token)
        print("Hugging Face token set successfully!")
    except Exception as e:
        print(f"Error setting Hugging Face token: {e}")

def list_available_cameras():
    """List all available camera indices."""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow to reduce noise on Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

class MetadataManager:
    def __init__(self, metadata_file: str = "object_metadata.json"):
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        # Add default metadata for common objects
        self.default_metadata = {
            "person": {
                "price": "N/A",
                "description": "Human being",
                "category": "Living Being",
                "details": {
                    "type": "Human",
                    "average_height": "170 cm",
                    "lifespan": "70-80 years",
                    "characteristics": ["Bipedal", "Intelligent", "Social"],
                    "biological_classification": "Homo sapiens"
                }
            },
            "man": {
                "price": "N/A",
                "description": "Adult male human",
                "category": "Living Being",
                "details": {
                    "type": "Human",
                    "gender": "Male",
                    "average_height": "175 cm",
                    "average_weight": "70 kg",
                    "biological_features": ["Facial hair", "Deeper voice", "Higher muscle mass"]
                }
            },
            "woman": {
                "price": "N/A",
                "description": "Adult female human",
                "category": "Living Being",
                "details": {
                    "type": "Human",
                    "gender": "Female",
                    "average_height": "165 cm",
                    "average_weight": "60 kg",
                    "biological_features": ["Higher body fat percentage", "Higher voice pitch"]
                }
            },
            "human face": {
                "price": "N/A",
                "description": "Frontal view of a human face",
                "category": "Body Part",
                "details": {
                    "type": "Facial Features",
                    "features": ["Eyes", "Nose", "Mouth", "Ears", "Eyebrows", "Cheeks", "Chin"],
                    "expressions": ["Happy", "Sad", "Angry", "Surprised", "Neutral"],
                    "sensory_organs": ["Eyes for vision", "Nose for smell", "Mouth for taste"]
                }
            },
            "glasses": {
                "price": "2000",
                "description": "Optical device worn on the face",
                "category": "Accessory",
                "details": {
                    "type": "Eyewear",
                    "material": "Metal/Plastic",
                    "purpose": "Vision correction/Style",
                    "variants": ["Reading glasses", "Sunglasses", "Computer glasses"],
                    "features": ["UV protection", "Anti-glare coating", "Lightweight frame"],
                    "maintenance": "Clean with microfiber cloth"
                }
            },
            "bottle": {
                "price": "100",
                "description": "Container for liquids",
                "category": "Container",
                "details": {
                    "type": "Storage",
                    "capacity": "500ml-1L",
                    "material": "Plastic/Glass",
                    "common_uses": ["Water", "Juice", "Soda", "Medicine"],
                    "recyclable": "Yes",
                    "durability": "Reusable"
                }
            },
            "cup": {
                "price": "150",
                "description": "Small container for drinking",
                "category": "Container",
                "details": {
                    "type": "Drinkware",
                    "capacity": "200-300ml",
                    "material": ["Ceramic", "Glass", "Plastic", "Metal"],
                    "common_uses": ["Hot beverages", "Cold drinks", "Water"],
                    "features": ["Handle", "Insulation", "Microwave safe"]
                }
            },
            "chair": {
                "price": "2500",
                "description": "Furniture for sitting",
                "category": "Furniture",
                "details": {
                    "type": "Seating",
                    "material": ["Wood", "Metal", "Plastic"],
                    "features": ["Backrest", "Armrests", "Cushion"],
                    "ergonomics": "Designed for comfort",
                    "common_uses": ["Office", "Dining", "Living room"]
                }
            },
            "laptop": {
                "price": "45000",
                "description": "Portable computer",
                "category": "Electronics",
                "details": {
                    "type": "Computer",
                    "components": ["Screen", "Keyboard", "Battery", "Processor"],
                    "features": ["Portable", "Wireless connectivity", "Built-in camera"],
                    "common_uses": ["Work", "Gaming", "Education"],
                    "power_source": "Battery/AC adapter"
                }
            },
            "cell phone": {
                "price": "15000",
                "description": "Mobile communication device",
                "category": "Electronics",
                "details": {
                    "type": "Smartphone",
                    "features": ["Touchscreen", "Camera", "GPS", "Internet"],
                    "connectivity": ["WiFi", "Bluetooth", "5G/4G"],
                    "applications": ["Communication", "Social media", "Navigation"],
                    "battery_life": "8-24 hours",
                    "storage": "64GB-1TB",
                    "display": "6-7 inches",
                    "camera": "12-108MP",
                    "processor": "Octa-core",
                    "os": ["Android", "iOS"]
                }
            },
            "book": {
                "price": "500",
                "description": "Collection of written or printed pages",
                "category": "Literature",
                "details": {
                    "type": "Reading material",
                    "format": ["Hardcover", "Paperback", "Digital"],
                    "content": ["Fiction", "Non-fiction", "Educational"],
                    "features": ["Pages", "Cover", "ISBN"],
                    "preservation": "Keep in dry place"
                }
            },
            "backpack": {
                "price": "1200",
                "description": "Bag carried on the back",
                "category": "Accessory",
                "details": {
                    "type": "Bag",
                    "material": ["Nylon", "Polyester", "Leather"],
                    "features": ["Multiple compartments", "Shoulder straps", "Zippers"],
                    "common_uses": ["School", "Travel", "Hiking"],
                    "capacity": "20-30 liters"
                }
            },
            "handbag": {
                "price": "3000",
                "description": "Small bag carried by hand",
                "category": "Accessory",
                "details": {
                    "type": "Bag",
                    "material": ["Leather", "Fabric", "Synthetic"],
                    "features": ["Handle", "Closure", "Interior pockets"],
                    "common_uses": ["Fashion", "Storage", "Organization"],
                    "styles": ["Tote", "Clutch", "Shoulder bag"]
                }
            },
            "keyboard": {
                "price": "2000",
                "description": "Input device for computers",
                "category": "Electronics",
                "details": {
                    "type": "Input device",
                    "features": ["Keys", "Numeric pad", "Function keys"],
                    "connectivity": ["USB", "Wireless", "Bluetooth"],
                    "types": ["Mechanical", "Membrane", "Gaming"],
                    "layout": "QWERTY"
                }
            },
            "mouse": {
                "price": "800",
                "description": "Pointing device for computers",
                "category": "Electronics",
                "details": {
                    "type": "Input device",
                    "features": ["Buttons", "Scroll wheel", "Optical sensor"],
                    "connectivity": ["USB", "Wireless", "Bluetooth"],
                    "types": ["Optical", "Laser", "Gaming"],
                    "ergonomics": "Designed for hand comfort"
                }
            },
            "tv": {
                "price": "25000",
                "description": "Television display device",
                "category": "Electronics",
                "details": {
                    "type": "Display",
                    "features": ["HD/4K resolution", "Smart TV", "Multiple ports"],
                    "screen_types": ["LED", "OLED", "QLED"],
                    "connectivity": ["HDMI", "USB", "WiFi"],
                    "sizes": "32-75 inches"
                }
            },
            "remote": {
                "price": "500",
                "description": "Control device for electronic equipment",
                "category": "Electronics",
                "details": {
                    "type": "Control device",
                    "features": ["Buttons", "Infrared sensor", "Battery powered"],
                    "compatibility": ["TV", "AC", "Set-top box"],
                    "battery_type": "AA/AAA",
                    "range": "Up to 10 meters"
                }
            },
            "clock": {
                "price": "1500",
                "description": "Time-keeping device",
                "category": "Timepiece",
                "details": {
                    "type": "Time device",
                    "display": ["Analog", "Digital"],
                    "power_source": ["Battery", "Electric", "Mechanical"],
                    "features": ["Alarm", "Date display", "Backlight"],
                    "placement": ["Wall", "Table", "Wrist"]
                }
            },
            "vase": {
                "price": "800",
                "description": "Decorative container for flowers",
                "category": "Decoration",
                "details": {
                    "type": "Container",
                    "material": ["Ceramic", "Glass", "Metal"],
                    "uses": ["Flower arrangement", "Decoration", "Storage"],
                    "styles": ["Modern", "Traditional", "Artistic"],
                    "maintenance": "Regular cleaning"
                }
            },
            "scissors": {
                "price": "200",
                "description": "Cutting tool with two blades",
                "category": "Tool",
                "details": {
                    "type": "Cutting tool",
                    "material": ["Stainless steel", "Plastic handles"],
                    "uses": ["Paper", "Fabric", "Hair"],
                    "features": ["Sharp blades", "Ergonomic handles"],
                    "safety": "Keep away from children"
                }
            },
            "toothbrush": {
                "price": "100",
                "description": "Oral hygiene tool",
                "category": "Hygiene",
                "details": {
                    "type": "Cleaning tool",
                    "features": ["Bristles", "Handle", "Head"],
                    "types": ["Manual", "Electric"],
                    "replacement": "Every 3 months",
                    "usage": "Twice daily"
                }
            },
            "mobile phone": {
                "price": "15000",
                "description": "Portable communication device",
                "category": "Electronics",
                "details": {
                    "type": "Smartphone",
                    "features": ["Touchscreen", "Camera", "GPS", "Internet"],
                    "connectivity": ["WiFi", "Bluetooth", "5G/4G"],
                    "applications": ["Communication", "Social media", "Navigation"],
                    "battery_life": "8-24 hours",
                    "storage": "64GB-1TB",
                    "display": "6-7 inches",
                    "camera": "12-108MP",
                    "processor": "Octa-core",
                    "os": ["Android", "iOS"]
                }
            },
            "smartphone": {
                "price": "15000",
                "description": "Advanced mobile phone with computing capabilities",
                "category": "Electronics",
                "details": {
                    "type": "Smartphone",
                    "features": ["Touchscreen", "Camera", "GPS", "Internet"],
                    "connectivity": ["WiFi", "Bluetooth", "5G/4G"],
                    "applications": ["Communication", "Social media", "Navigation"],
                    "battery_life": "8-24 hours",
                    "storage": "64GB-1TB",
                    "display": "6-7 inches",
                    "camera": "12-108MP",
                    "processor": "Octa-core",
                    "os": ["Android", "iOS"]
                }
            },
            "phone": {
                "price": "15000",
                "description": "Communication device",
                "category": "Electronics",
                "details": {
                    "type": "Smartphone",
                    "features": ["Touchscreen", "Camera", "GPS", "Internet"],
                    "connectivity": ["WiFi", "Bluetooth", "5G/4G"],
                    "applications": ["Communication", "Social media", "Navigation"],
                    "battery_life": "8-24 hours",
                    "storage": "64GB-1TB",
                    "display": "6-7 inches",
                    "camera": "12-108MP",
                    "processor": "Octa-core",
                    "os": ["Android", "iOS"]
                }
            },
            "tablet": {
                "price": "25000",
                "description": "Portable touchscreen computer",
                "category": "Electronics",
                "details": {
                    "type": "Computing device",
                    "features": ["Touchscreen", "Camera", "GPS", "Internet"],
                    "connectivity": ["WiFi", "Bluetooth", "5G/4G"],
                    "applications": ["Reading", "Gaming", "Productivity"],
                    "battery_life": "8-12 hours",
                    "storage": "64GB-1TB",
                    "display": "8-12 inches",
                    "camera": "8-12MP",
                    "processor": "Octa-core",
                    "os": ["Android", "iOS", "Windows"]
                }
            },
            "headphones": {
                "price": "2000",
                "description": "Audio listening device",
                "category": "Electronics",
                "details": {
                    "type": "Audio device",
                    "features": ["Noise cancellation", "Bluetooth", "Microphone"],
                    "connectivity": ["Wireless", "Wired", "Bluetooth"],
                    "types": ["Over-ear", "In-ear", "On-ear"],
                    "battery_life": "20-30 hours",
                    "sound_quality": "High-fidelity",
                    "comfort": "Ergonomic design"
                }
            },
            "earphones": {
                "price": "1500",
                "description": "Small audio listening device",
                "category": "Electronics",
                "details": {
                    "type": "Audio device",
                    "features": ["Noise isolation", "Bluetooth", "Microphone"],
                    "connectivity": ["Wireless", "Wired", "Bluetooth"],
                    "types": ["In-ear", "True wireless", "Sports"],
                    "battery_life": "4-8 hours",
                    "sound_quality": "High-fidelity",
                    "comfort": "Ergonomic fit"
                }
            },
            "speaker": {
                "price": "3000",
                "description": "Audio output device",
                "category": "Electronics",
                "details": {
                    "type": "Audio device",
                    "features": ["Bluetooth", "Waterproof", "Bass"],
                    "connectivity": ["Wireless", "Aux", "Bluetooth"],
                    "types": ["Portable", "Smart", "Party"],
                    "battery_life": "10-20 hours",
                    "sound_quality": "360° sound",
                    "power": "10-40W"
                }
            },
            "camera": {
                "price": "35000",
                "description": "Image capture device",
                "category": "Electronics",
                "details": {
                    "type": "Imaging device",
                    "features": ["Auto-focus", "Image stabilization", "Flash"],
                    "types": ["DSLR", "Mirrorless", "Point-and-shoot"],
                    "resolution": "12-50MP",
                    "lens": ["Interchangeable", "Fixed"],
                    "storage": "SD card",
                    "connectivity": ["WiFi", "Bluetooth", "USB"]
                }
            },
            "watch": {
                "price": "5000",
                "description": "Time-keeping device worn on wrist",
                "category": "Timepiece",
                "details": {
                    "type": "Wristwatch",
                    "features": ["Water resistant", "Chronograph", "Date display"],
                    "types": ["Analog", "Digital", "Smart"],
                    "material": ["Stainless steel", "Leather", "Rubber"],
                    "movement": ["Quartz", "Automatic", "Smart"],
                    "battery_life": "1-2 years",
                    "style": ["Casual", "Sports", "Luxury"]
                }
            },
            "sunglasses": {
                "price": "1500",
                "description": "Protective eyewear for sun",
                "category": "Accessory",
                "details": {
                    "type": "Eyewear",
                    "features": ["UV protection", "Polarized", "Anti-glare"],
                    "material": ["Plastic", "Metal", "Glass"],
                    "styles": ["Aviator", "Wayfarer", "Sports"],
                    "lens_color": ["Black", "Brown", "Gradient"],
                    "protection": "100% UV",
                    "durability": "Impact resistant"
                }
            },
            "umbrella": {
                "price": "800",
                "description": "Rain protection device",
                "category": "Accessory",
                "details": {
                    "type": "Protection",
                    "features": ["Waterproof", "Wind resistant", "Auto-open"],
                    "material": ["Polyester", "Nylon", "Metal frame"],
                    "size": ["Compact", "Standard", "Golf"],
                    "mechanism": ["Manual", "Automatic"],
                    "protection": "Rain and UV",
                    "portability": "Foldable"
                }
            },
            "wallet": {
                "price": "1200",
                "description": "Card and money holder",
                "category": "Accessory",
                "details": {
                    "type": "Storage",
                    "features": ["Card slots", "Coin pocket", "ID window"],
                    "material": ["Leather", "Synthetic", "Fabric"],
                    "styles": ["Bifold", "Trifold", "Slim"],
                    "capacity": "6-12 cards",
                    "security": "RFID blocking",
                    "durability": "Long-lasting"
                }
            },
            "pen": {
                "price": "50",
                "description": "Writing instrument",
                "category": "Stationery",
                "details": {
                    "type": "Writing tool",
                    "features": ["Retractable", "Grip", "Clip"],
                    "types": ["Ballpoint", "Gel", "Fountain"],
                    "ink_color": ["Blue", "Black", "Red"],
                    "material": ["Plastic", "Metal", "Wood"],
                    "refillable": "Yes",
                    "grip": "Ergonomic"
                }
            },
            "pencil": {
                "price": "20",
                "description": "Writing instrument with graphite core",
                "category": "Stationery",
                "details": {
                    "type": "Writing tool",
                    "features": ["Erasable", "Sharpener", "Eraser"],
                    "types": ["Wooden", "Mechanical", "Colored"],
                    "lead_size": ["0.5mm", "0.7mm", "2mm"],
                    "material": ["Wood", "Plastic", "Metal"],
                    "erasable": "Yes",
                    "durability": "Break-resistant"
                }
            }
        }
        
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    print(f"Successfully loaded metadata for {len(metadata)} objects")
                    return metadata
            else:
                print(f"Metadata file {self.metadata_file} not found")
        except Exception as e:
            print(f"Error loading metadata file: {e}")
        return {}
    
    def get_object_metadata(self, class_name: str) -> Dict:
        """Get metadata for a specific object class."""
        # First check the loaded metadata
        if class_name in self.metadata:
            return self.metadata[class_name]
        # Then check default metadata (case-insensitive)
        class_name_lower = class_name.lower()
        if class_name_lower in self.default_metadata:
            return self.default_metadata[class_name_lower]
        # Return default values if no metadata found
        return {
            "price": "N/A",
            "description": f"A {class_name}",
            "category": "Uncategorized",
            "details": {
                "type": class_name,
                "status": "Unknown"
            }
        }

class FAISSSearch:
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        
    def add_vector(self, vector: np.ndarray, metadata: Dict):
        """Add a vector and its metadata to the index."""
        self.index.add(vector.reshape(1, -1))
        self.metadata.append(metadata)
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar vectors."""
        distances, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append({
                    "metadata": self.metadata[idx],
                    "distance": float(distances[0][i])
                })
        return results

def get_device(device_str: str) -> str:
    """Get the appropriate device string based on availability."""
    if device_str.lower() == 'cuda':
        if torch.cuda.is_available():
            print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        print("CUDA is not available. Using CPU.")
    return 'cpu'

class ObjectDetector:
    def __init__(
        self,
        yolo_weights: str = str(config.YOLO_WEIGHTS),
        sam_weights: str = str(config.SAM_WEIGHTS),
        device: str = config.DEVICE,
        conf_threshold: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        enable_embeddings: bool = True,
        img_size: int = 640,
        use_fp16: bool = True,
        hf_token: Optional[str] = None,
        use_fast: bool = True
    ):
        # Set Hugging Face token if provided
        if hf_token:
            set_hf_token(hf_token)
        
        # Get appropriate device
        self.device = torch.device(get_device(device))
        print(f"Using device: {self.device}")
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        
        print(f"Initializing YOLO model on {self.device}...")
        # Initialize YOLO model with optimizations
        self.yolo_model = YOLO(yolo_weights)
        
        # Move model to device
        try:
            self.yolo_model.to(self.device)
            print(f"YOLO model moved to {self.device}")
        except Exception as e:
            print(f"Error moving YOLO model to device: {e}")
            print("Falling back to CPU for YOLO model")
            self.device = torch.device('cpu')
            self.yolo_model.to(self.device)
        
        # Enable FP16 if available and requested
        if use_fp16 and self.device.type == 'cuda':
            try:
                self.yolo_model.model.half()
                print("FP16 optimization enabled")
            except Exception as e:
                print(f"FP16 optimization not available: {e}")
        
        # Set model to evaluation mode
        self.yolo_model.model.eval()
        
        print(f"Initializing SAM model on {self.device}...")
        # Initialize SAM model with optimizations
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_weights)
        try:
            self.sam = self.sam.to(device=self.device)
            if use_fp16 and self.device.type == 'cuda':
                self.sam = self.sam.half()
                print("SAM FP16 optimization enabled")
        except Exception as e:
            print(f"Error moving SAM model to device: {e}")
            print("Falling back to CPU for SAM model")
            self.device = torch.device('cpu')
        self.sam = self.sam.to(device=self.device)
        
        self.sam_predictor = SamPredictor(self.sam)
        
        # Initialize embedding manager if enabled
        self.embedding_manager = None
        if enable_embeddings:
            print("Initializing embedding manager...")
            self.embedding_manager = EmbeddingManager(device=str(self.device))
            self.embedding_manager.load_index()
        
        # Load COCO class names
        self.class_names = self.yolo_model.names
        print("Initialization complete!")
        
        self.metadata_manager = MetadataManager()
        self.faiss_search = FAISSSearch()
        self.running = True  # Flag for quit functionality
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for faster inference"""
        # Resize frame while maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized

    def stop(self):
        """Stop the detector."""
        self.running = False
    
    def detect(
        self,
        image: np.ndarray,
        show_boxes: bool = config.SHOW_BOXES,
        show_labels: bool = config.SHOW_LABELS,
        show_confidence: bool = config.SHOW_CONFIDENCE,
        show_segmentation: bool = config.SHOW_SEGMENTATION,
        save_embeddings: bool = True
    ) -> Tuple[np.ndarray, List[np.ndarray], List[Dict]]:
        """Perform detection, segmentation, and embedding on an image."""

        # Preprocess image for faster inference
        processed_image = self.preprocess_frame(image)
        
        # Dictionary to store object counts
        object_counts = {}
        
        # YOLO detection with optimized settings
        with torch.no_grad():  # Disable gradient calculation
            try:
                results = self.yolo_model(
                    processed_image,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )[0]

                # Convert results to numpy arrays
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                # Create detections object
        detections = sv.Detections(
                    xyxy=boxes,
                    confidence=scores,
                    class_id=class_ids
        )

                # Scale detections back to original image size
                h, w = image.shape[:2]
                scale = min(self.img_size / w, self.img_size / h)
                detections.xyxy = detections.xyxy / scale
                
                # Draw detections
                annotated_image = image.copy()
                if show_boxes:
                    # Count objects
                    for cls_id in detections.class_id:
                        class_name = self.class_names[cls_id]
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    # Print total counts
                    print("\n" + "="*50)
                    print("Object Counts:")
                    for obj_name, count in object_counts.items():
                        print(f"{obj_name}: {count}")
                    print("="*50 + "\n")
                    
                    for i in range(len(detections)):
                        box = detections.xyxy[i]
                        conf = detections.confidence[i]
                        cls_id = detections.class_id[i]
                        
                        # Get class name
                        class_name = self.class_names[cls_id]
                        
                        # Get metadata
                        object_info = self.metadata_manager.get_object_metadata(class_name)
                        price = object_info.get('price', 'N/A')
                        description = object_info.get('description', '')
                        category = object_info.get('category', '')
                        details = object_info.get('details', {})
                        
                        # Format price with ₹ symbol
                        if price != 'N/A':
                            try:
                                price_int = int(price)
                                price_text = f"₹{price_int:,}"
                            except ValueError:
                                price_text = "Price: N/A"
                        else:
                            price_text = "Price: N/A"
                        
                        # Create label with count, confidence, price and description
                        count = object_counts[class_name]
                        label = f"{class_name} ({count}) - {conf:.2f}\n{price_text}\n{description}"
                        
                        # Print detailed information to terminal
                        print("\n" + "="*50)
                        print(f"Detected: {class_name} ({count})")
                        print(f"Confidence: {conf:.2f}")
                        print(f"Price: {price_text}")
                        print(f"Category: {category}")
                        print(f"Description: {description}")
                        if details:
                            print("\nDetails:")
                            for key, value in details.items():
                                if isinstance(value, list):
                                    value = ', '.join(value)
                                print(f"  {key}: {value}")
                        print("="*50 + "\n")
                        
                        # Draw box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label background with more space for description
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        # Add extra height for description
                        label_height = int(label_height * 2.5)  # Adjust multiplier based on description length
                        cv2.rectangle(annotated_image,
                                    (x1, y1 - label_height - 10),
                                    (x1 + label_width, y1),
                                    (0, 255, 0), -1)

                        # Draw label text with description
                        y_text = y1 - 5
                        for i, line in enumerate(label.split('\n')):
                            cv2.putText(annotated_image, line,
                                       (x1, y_text - (i * 20)),  # Adjust line spacing
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        # Draw additional details if available
                        if details:
                            details_text = []
                            for key, value in details.items():
                                if isinstance(value, list):
                                    value = ', '.join(value)
                                details_text.append(f"{key}: {value}")
                            
                            details_label = '\n'.join(details_text)
                            y_offset = int(y1 - label_height - 10)
                            
                            # Draw details background
                            (details_width, details_height), _ = cv2.getTextSize(details_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                            cv2.rectangle(annotated_image,
                                        (x1, y_offset - details_height - 5),
                                        (x1 + details_width, y_offset),
                                        (0, 200, 0), -1)
                            
                            # Draw details text
                            cv2.putText(annotated_image, details_label,
                                       (x1, y_offset - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                return annotated_image, [], []
                
            except Exception as e:
                print(f"Error during YOLO detection: {e}")
                return image, [], []

def main():
    parser = argparse.ArgumentParser(description='Object Detection with YOLO and SAM')
    parser.add_argument('--source', type=str, default='0', help='Video source (camera index or video file)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--hf-token', type=str, help='Hugging Face token')
    args = parser.parse_args()
    
    # Initialize detector with GPU if available
    detector = ObjectDetector(
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
        hf_token=args.hf_token
    )
    
    # Handle video source
    if args.source.isdigit():
        source = int(args.source)
    else:
    source = args.source

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
        # Process frame
        result_frame, masks, detections = detector.detect(frame)
                
                # Display result
        cv2.imshow('Object Detection', result_frame)
                
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
                    break
            
                cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 