#!/usr/bin/env python3
"""
User Input Test Script for Multi-Intent NLP Model
Modify the test_sentences list with your own inputs
"""

def test_with_user_inputs():
    """Test the model with custom user inputs"""
    print("🤖 Multi-Intent NLP Model - User Input Test")
    print("=" * 50)
    
    try:
        from inference_example import MultiIntentPredictor
        
        # Initialize predictor
        print("📥 Loading model...")
        predictor = MultiIntentPredictor()
        print("✅ Model loaded successfully!")
        
        # ADD YOUR TEST SENTENCES HERE
        test_sentences = [
            "I want to book a flight to Paris",
            "Can you help me cancel my hotel reservation?",
            "What are your customer support hours?",
            "I need to change my booking details",
            "The product I received is damaged and I want a refund",
            "Please confirm my order and send payment instructions",
            "How do I track my shipment?",
            "I have a question about my invoice"
        ]
        
        print(f"🧪 Testing with {len(test_sentences)} custom sentences...")
        print("=" * 50)
        
        for i, sentence in enumerate(test_sentences, 1):
            print(f"\nTest {i}: '{sentence}'")
            print("-" * 40)
            
            # Get predictions
            results = predictor.predict(sentence, threshold=0.3)
            probabilities = predictor.predict_proba(sentence)
            
            if results:
                print("🎯 PREDICTED INTENTS:")
                for result in results:
                    print(f"   ✅ {result['intent']} (confidence: {result['confidence']:.3f})")
            else:
                print("❌ No intents detected above threshold")
            
            print("\n📈 ALL PROBABILITIES:")
            for intent, score in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1:  # Only show probabilities above 10%
                    marker = "🎯" if score >= 0.3 else "  "
                    print(f"   {marker} {intent}: {score:.3f}")
            
            print("-" * 40)
        
        print("\n🎉 Testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run: pip install -r requirements.txt first")

if __name__ == "__main__":
    test_with_user_inputs()
