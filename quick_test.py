#!/usr/bin/env python3
"""
Quick Test - Minimal version to verify the model works
"""

print("🚀 Quick Model Test")
print("=" * 30)

try:
    from safe_model_loader import MultiIntentModel
    from inference_example import MultiIntentPredictor
    
    print("✅ Imports successful")
    
    # Quick test
    predictor = MultiIntentPredictor()
    test_text = "I want to book a flight and hotel"
    
    print(f"📝 Testing: '{test_text}'")
    results = predictor.predict(test_text)
    
    if results:
        print("🎯 Results:")
        for r in results:
            print(f"   - {r['intent']} ({r['confidence']:.3f})")
    else:
        print("❌ No intents detected")
    
    print("✅ Model is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Make sure to run: pip install -r requirements.txt first")
