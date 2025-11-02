#!/usr/bin/env python3
"""
Example usage of the Multi-Intent NLP Model
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from model_loader import MultiIntentModel
    
    print("🚀 Loading Multi-Intent Model...")
    
    # Initialize and load the model
    model_loader = MultiIntentModel()
    model = model_loader.load()
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model)}")
    
    # You can now use the model for inference
    # Add your inference code here
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Make sure you have installed the requirements:")
    print("   pip install -r requirements.txt")
