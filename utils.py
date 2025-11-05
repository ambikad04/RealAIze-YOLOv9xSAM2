import cv2
import numpy as np
import torch
from typing import Tuple, List, Dict, Any
import supervision as sv

def load_image(image_path: str) -> np.ndarray:
    """Load an image from path."""
    return cv2.imread(image_path)

def preprocess_image(image: np.ndarray, size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for model input."""
    # Get original image dimensions
    height, width = image.shape[:2]
    
    # Resize image
    resized = cv2.resize(image, (size, size))
    
    # Convert to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to tensor
    tensor = torch.from_numpy(rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    
    return tensor, (height, width)

def postprocess_detections(
    predictions: torch.Tensor,
    original_size: Tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict[str, Any]]:
    """Convert model predictions to detection format."""
    # Convert predictions to numpy
    pred_np = predictions.cpu().numpy()
    
    # Filter by confidence
    mask = pred_np[:, 4] > conf_threshold
    filtered_preds = pred_np[mask]
    
    # Convert to supervision format
    detections = sv.Detections(
        xyxy=filtered_preds[:, :4],
        confidence=filtered_preds[:, 4],
        class_id=filtered_preds[:, 5].astype(int)
    )
    
    # Apply NMS
    detections = detections.with_nms(iou_threshold)
    
    return detections

def draw_detections(
    image: np.ndarray,
    detections: sv.Detections,
    masks: List[np.ndarray] = None,
    class_names: List[str] = None,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_confidence: bool = True,
    show_segmentation: bool = True
) -> np.ndarray:
    """Draw detections on image."""
    annotated_image = image.copy()
    
    if show_boxes:
        # Draw bounding boxes
        for xyxy, confidence, class_id in detections:
            # Draw box
            cv2.rectangle(
                annotated_image,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                (0, 255, 0),
                2
            )
            
            if show_labels or show_confidence:
                # Prepare label text
                label = []
                if show_labels and class_names:
                    label.append(class_names[class_id])
                if show_confidence:
                    label.append(f"{confidence:.2f}")
                
                # Draw label
                text = " ".join(label)
                cv2.putText(
                    annotated_image,
                    text,
                    (int(xyxy[0]), int(xyxy[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
    
    if show_segmentation and masks:
        # Draw segmentation masks
        for mask in masks:
            colored_mask = np.zeros_like(annotated_image)
            colored_mask[mask] = (0, 255, 0)
            annotated_image = cv2.addWeighted(
                annotated_image,
                1,
                colored_mask,
                0.5,
                0
            )
    
    return annotated_image

def get_device() -> torch.device:
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") 