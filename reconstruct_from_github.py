#!/usr/bin/env python3
import os
import requests
import base64
import json

def download_chunk_from_github(chunk_number):
    """Download a single chunk from GitHub"""
    url = f"https://raw.githubusercontent.com/HariGW-0/multi-intent-nlp-model/main/model_chunks/model_chunk_{chunk_number:03d}.bin"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            print(f"Failed to download chunk {chunk_number}")
            return None
    except Exception as e:
        print(f"Error downloading chunk {chunk_number}: {e}")
        return None

def reconstruct_model():
    """Reconstruct the model by downloading all chunks"""
    print("🔧 Reconstructing model from GitHub chunks...")
    
    total_chunks = 84
    all_data = b''
    
    for i in range(1, total_chunks + 1):
        print(f"📥 Downloading chunk {i}/{total_chunks}...")
        chunk_data = download_chunk_from_github(i)
        if chunk_data is None:
            print("❌ Failed to reconstruct model")
            return False
        
        all_data += chunk_data
        chunk_size_mb = len(chunk_data) / (1024 * 1024)
        print(f"✅ Chunk {i} downloaded ({chunk_size_mb:.2f} MB)")
    
    # Save reconstructed model
    with open("multi_intent_model_reconstructed.pth", "wb") as f:
        f.write(all_data)
    
    print(f"🎉 Model reconstructed successfully!")
    print(f"📊 Final size: {len(all_data)/(1024*1024):.2f} MB")
    return True

if __name__ == "__main__":
    reconstruct_model()
