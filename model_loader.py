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
        
        # Load the model with proper settings for PyTorch 2.6+
        print("📥 Loading model into memory...")
        try:
            # First try with weights_only=False (for PyTorch 2.6+)
            self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"⚠️ Standard load failed: {e}")
            print("🔄 Trying alternative loading method...")
            try:
                # Alternative method for older PyTorch versions
                self.model = torch.load(model_path, map_location='cpu')
            except Exception as e2:
                print(f"❌ All loading methods failed: {e2}")
                raise Exception("Could not load model file")
        
        print("✅ Model loaded successfully!")
        return self.model

# Usage example
if __name__ == "__main__":
    loader = MultiIntentModel()
    model = loader.load()
    print("Model ready for inference!")
