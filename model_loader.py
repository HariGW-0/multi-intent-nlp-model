import torch
import os
from reconstruct_from_github import reconstruct_model

class MultiIntentModel:
    def __init__(self):
        self.model = None
    
    def load(self):
        """Load the model, reconstructing if necessary"""
        model_path = "multi_intent_model_reconstructed.pth"
        
        # Reconstruct if needed
        if not os.path.exists(model_path):
            print("📦 Model not found. Reconstructing from GitHub chunks...")
            if not reconstruct_model():
                raise Exception("Failed to reconstruct model")
        
        # Load the model
        print("📥 Loading model into memory...")
        self.model = torch.load(model_path, map_location='cpu')
        print("✅ Model loaded successfully!")
        return self.model

# Usage example
if __name__ == "__main__":
    loader = MultiIntentModel()
    model = loader.load()
    print("Model ready for inference!")
