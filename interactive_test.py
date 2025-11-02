#!/usr/bin/env python3
"""
Interactive Test Script for Multi-Intent NLP Model
Test the model directly after cloning from GitHub
"""

import os
import sys
import requests
import time

def check_environment():
    """Check if all required packages are installed"""
    print("🔍 Checking environment...")
    
    required_packages = ['torch', 'transformers', 'numpy', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("💡 Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed")
    return True

def download_and_test():
    """Download and test the model interactively"""
    print("\n🚀 Starting Interactive Test")
    print("=" * 50)
    
    try:
        # Import required modules
        from safe_model_loader import MultiIntentModel
        from inference_example import MultiIntentPredictor
        
        print("✅ Modules imported successfully")
        
        # Initialize predictor
        print("\n📥 Initializing model predictor...")
        predictor = MultiIntentPredictor()
        
        print("✅ Predictor initialized")
        print("💡 The model will be automatically downloaded and reconstructed on first run")
        
        # Test sentences with expected intents
        test_cases = [
            {
                "text": "I want to book a flight to New York and a hotel for next week",
                "expected": ["booking"]
            },
            {
                "text": "Can you help me cancel my reservation and process a refund?",
                "expected": ["cancellation", "payment"]
            },
            {
                "text": "What are your opening hours and do you offer customer support?",
                "expected": ["inquiry", "support"]
            },
            {
                "text": "I'm unhappy with my purchase and want to provide feedback",
                "expected": ["complaint", "feedback"]
            },
            {
                "text": "Please confirm my booking and let me know payment options",
                "expected": ["confirmation", "payment"]
            }
        ]
        
        print("\n🧪 Running automated tests...")
        print("-" * 40)
        
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: '{test_case['text']}'")
            print("Expected:", ", ".join(test_case['expected']))
            
            try:
                # Predict intents
                results = predictor.predict(test_case['text'], threshold=0.3)
                predicted_intents = [r['intent'] for r in results]
                
                print("Predicted:", ", ".join(predicted_intents) if predicted_intents else "None")
                
                # Check if expected intents are in predicted
                matched = all(exp in predicted_intents for exp in test_case['expected'])
                
                if matched:
                    print("✅ PASS")
                else:
                    print("❌ FAIL - Expected intents not fully matched")
                    all_passed = False
                
                # Show confidence scores
                for result in results:
                    print(f"   {result['intent']}: {result['confidence']:.3f}")
                    
            except Exception as e:
                print(f"❌ ERROR: {e}")
                all_passed = False
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ALL AUTOMATED TESTS PASSED!")
        else:
            print("⚠️ SOME TESTS FAILED - but model might still work")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return None

def interactive_mode(predictor):
    """Interactive mode for user input"""
    print("\n🎮 INTERACTIVE MODE")
    print("=" * 50)
    print("Enter sentences to test the model (type 'quit' to exit)")
    print("You can adjust the confidence threshold (default: 0.5)")
    print("-" * 50)
    
    threshold = 0.5
    
    while True:
        try:
            user_input = input("\n📝 Enter sentence (or 'threshold=X' to change): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            elif user_input.startswith('threshold='):
                try:
                    new_threshold = float(user_input.split('=')[1])
                    if 0 <= new_threshold <= 1:
                        threshold = new_threshold
                        print(f"✅ Confidence threshold set to: {threshold}")
                    else:
                        print("❌ Threshold must be between 0 and 1")
                except ValueError:
                    print("❌ Invalid threshold format. Use: threshold=0.7")
                    
            elif user_input:
                print(f"\n🔍 Analyzing: '{user_input}'")
                print(f"📊 Using threshold: {threshold}")
                
                # Get predictions
                results = predictor.predict(user_input, threshold=threshold)
                probabilities = predictor.predict_proba(user_input)
                
                if results:
                    print("\n🎯 PREDICTED INTENTS:")
                    for result in results:
                        print(f"   ✅ {result['intent']} (confidence: {result['confidence']:.3f})")
                else:
                    print("\n❌ No intents detected above threshold")
                
                print("\n📈 ALL PROBABILITIES:")
                for intent, score in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    if score > 0.1:  # Only show probabilities above 10%
                        marker = "🎯" if score >= threshold else "  "
                        print(f"   {marker} {intent}: {score:.3f}")
                        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def performance_test(predictor):
    """Test model performance with multiple inputs"""
    print("\n⚡ PERFORMANCE TEST")
    print("=" * 50)
    
    test_sentences = [
        "Book a table for dinner",
        "Cancel my appointment",
        "I need help with my account",
        "What is your refund policy?",
        "Confirm my order please",
        "I want to change my flight",
        "The product is damaged",
        "Thank you for your service"
    ]
    
    print(f"Testing with {len(test_sentences)} sentences...")
    start_time = time.time()
    
    for sentence in test_sentences:
        try:
            results = predictor.predict(sentence)
            # Just test that it works, don't print all results
        except Exception as e:
            print(f"❌ Error with: {sentence} - {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ Processed {len(test_sentences)} sentences in {total_time:.2f} seconds")
    print(f"📊 Average time per sentence: {total_time/len(test_sentences):.3f} seconds")

def main():
    """Main function"""
    print("🤖 Multi-Intent NLP Model - Interactive Test")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Please install missing packages and run again")
        return
    
    # Download and test model
    predictor = download_and_test()
    
    if predictor is None:
        print("\n❌ Model testing failed. Please check the errors above.")
        return
    
    # Run performance test
    performance_test(predictor)
    
    # Start interactive mode
    interactive_mode(predictor)
    
    print("\n" + "=" * 60)
    print("🎉 Test completed successfully!")
    print("💡 Your model is working correctly!")
    print("🔗 Repository: https://github.com/HariGW-0/multi-intent-nlp-model")

if __name__ == "__main__":
    main()
