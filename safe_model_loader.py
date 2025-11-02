import torch
import os
import warnings

def safe_torch_load(model_path, map_location='cpu'):
    """Safely load PyTorch model with compatibility for different versions"""
    
    # Try different loading methods
    methods = [
        # Method 1: PyTorch 2.6+ with weights_only=False
        lambda: torch.load(model_path, map_location=map_location, weights_only=False),
        
        # Method 2: Legacy loading (for PyTorch < 2.6)
        lambda: torch.load(model_path, map_location=map_location),
        
        # Method 3: With safe globals for specific transformers objects
        lambda: torch.load(model_path, map_location=map_location, weights_only=True)
    ]
    
    errors = []
    for i, method in enumerate(methods):
        try:
            print(f"🔄 Trying loading method {i+1}...")
            model = method()
            print(f"✅ Success with method {i+1}")
            return model
        except Exception as e:
            errors.append(f"Method {i+1}: {str(e)}")
            continue
    
    # If all methods fail, raise comprehensive error
    error_msg = "All loading methods failed:\n" + "\n".join(errors)
    raise RuntimeError(error_msg)

class MultiIntentModel:
    def __init__(self):
        self.model = None
    
    def load(self):
        """Load the model, reconstructing if necessary"""
        from reconstruct_from_github import reconstruct_model
        
        model_path = "multi_intent_model_reconstructed.pth"
        
        # Reconstruct if needed
        if not os.path.exists(model_path):
            print("📦 Model not found. Reconstructing from GitHub chunks...")
            if not reconstruct_model():
                raise Exception("Failed to reconstruct model")
        
        # Load the model safely
        print("📥 Loading model into memory...")
        self.model = safe_torch_load(model_path)
        print("✅ Model loaded successfully!")
        return self.model

# Usage example
if __name__ == "__main__":
    loader = MultiIntentModel()
    model = loader.load()
    print("Model ready for inference!")
