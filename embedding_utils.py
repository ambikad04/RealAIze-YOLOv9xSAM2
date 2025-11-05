import torch
import numpy as np
import faiss
from typing import List, Tuple, Dict
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from pathlib import Path
import json
from huggingface_hub import HfApi, login
import os

# Try to get token from environment variable
token = os.getenv("HUGGINGFACE_TOKEN")

# Only attempt login if token is available
if token:
    try:
        login(token=token)
        print("Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"Warning: Could not login to Hugging Face: {e}")
        print("Continuing without Hugging Face authentication...")
else:
    print("No Hugging Face token found. Some features may be limited.")
    print("To enable full functionality, set the HUGGINGFACE_TOKEN environment variable.")


class EmbeddingManager:
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        index_path: str = "embeddings/faiss_index.bin",
        metadata_path: str = "embeddings/metadata.json"
    ):
        self.device = torch.device(device)
        self.model_name = model_name
        
        # Initialize DINOv2 model
        print("Loading DINOv2 model...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize FAISS index
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create or load FAISS index
        self.dimension = 768  # DINOv2 base model dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = self._load_metadata()
        print("Embedding manager initialized successfully!")
    
    def _load_metadata(self) -> Dict:
        """Load metadata if exists, otherwise create new."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {"objects": []}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Get embedding for an image using DINOv2."""
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                else:
                    embedding = outputs.cpu().numpy()
            
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return zero embedding as fallback
            return np.zeros((1, self.dimension))
    
    def add_to_index(
        self,
        image: np.ndarray,
        object_id: str,
        class_name: str,
        confidence: float,
        bbox: List[float]
    ):
        """Add object embedding to FAISS index."""
        try:
            # Get embedding
            embedding = self.get_embedding(image)
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Update metadata
            self.metadata["objects"].append({
                "id": object_id,
                "class": class_name,
                "confidence": confidence,
                "bbox": bbox,
                "index": len(self.metadata["objects"])
            })
            
            # Save metadata
            self._save_metadata()
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
        except Exception as e:
            print(f"Error adding to index: {e}")
    
    def search_similar(
        self,
        image: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        """Search for similar objects in the index."""
        try:
            # Get query embedding
            query_embedding = self.get_embedding(image)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get results
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.metadata["objects"]):  # Valid index
                    obj_data = self.metadata["objects"][idx]
                    results.append({
                        **obj_data,
                        "distance": float(dist)
                    })
            
            return results
        except Exception as e:
            print(f"Error searching similar objects: {e}")
            return []
    
    def load_index(self):
        """Load existing FAISS index if available."""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.metadata = self._load_metadata()
                print("Loaded existing index successfully!")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Create new index if loading fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {"objects": []} 

            