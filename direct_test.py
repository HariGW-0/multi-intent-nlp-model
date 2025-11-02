#!/usr/bin/env python3
"""
Direct Test - Simple script for quick model testing
"""

def main():
    print("🚀 Direct Model Test")
    print("=" * 40)
    
    try:
        from inference_example import MultiIntentPredictor
        
        print("📥 Loading model...")
        predictor = MultiIntentPredictor()
        
        print("✅ Model ready! Enter sentences to test.")
        print("   Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            sentence = input("\n🎯 Enter sentence: ").strip()
            
            if sentence.lower() in ['quit', 'exit', 'q']:
                break
                
            if sentence:
                results = predictor.predict(sentence)
                
                if results:
                    print("📊 Results:")
                    for r in results:
                        print(f"   - {r['intent']} ({r['confidence']:.3f})")
                else:
                    print("   ❌ No intents detected")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
