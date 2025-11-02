#!/usr/bin/env python3
"""
Complete inference example for Multi-Intent NLP Model
"""
import torch
from transformers import AutoTokenizer
from model_architecture import MultiIntentClassifier
from safe_model_loader import MultiIntentModel
import numpy as np

class MultiIntentPredictor:
    def __init__(self, num_intents=10):
        self.num_intents = num_intents
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.intent_labels = [
            "booking", "inquiry", "complaint", "support", "feedback",
            "payment", "cancellation", "modification", "confirmation", "other"
        ]
    
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            loader = MultiIntentModel()
            loaded_data = loader.load()
            
            # Check if loaded_data is a model or a state dict
            if isinstance(loaded_data, torch.nn.Module):
                self.model = loaded_data
            else:
                # If it's a state dict, create model and load weights
                self.model = MultiIntentClassifier(num_intents=self.num_intents)
                self.model.load_state_dict(loaded_data)
            
            self.model.eval()
        return self.model
    
    def predict(self, text, threshold=0.5):
        """Predict intents for given text"""
        self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold).int()
        
        # Convert to readable format
        results = []
        for i in range(self.num_intents):
            if predictions[0][i] == 1:
                results.append({
                    'intent': self.intent_labels[i],
                    'confidence': probabilities[0][i].item(),
                    'label_index': i
                })
        
        return results
    
    def predict_proba(self, text):
        """Get probability scores for all intents"""
        self.load_model()
        
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs)
        
        return {
            self.intent_labels[i]: probabilities[0][i].item()
            for i in range(self.num_intents)
        }

# Example usage
if __name__ == "__main__":
    print("🚀 Multi-Intent NLP Model Inference Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = MultiIntentPredictor()
    
    # Test sentences
    test_sentences = [
        "I want to book a flight and hotel for my vacation",
        "My order hasn't arrived yet, can you help?",
        "I need to cancel my reservation and get a refund",
        "How do I change my booking details?"
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 Sentence: '{sentence}'")
        print("🔍 Predicted Intents:")
        
        results = predictor.predict(sentence)
        if results:
            for result in results:
                print(f"   ✅ {result['intent']} (confidence: {result['confidence']:.3f})")
        else:
            print("   ❌ No intents detected above threshold")
        
        print("-" * 40)
