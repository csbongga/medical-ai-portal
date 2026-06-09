#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Image Prediction Endpoint
"""
import requests
import json
import sys
import os

# Fix Windows console encoding for Thai text
if sys.platform.startswith('win'):
    try:
        os.system('chcp 65001 > nul')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def safe_print(text):
    """Print text with proper encoding handling"""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoded = text.encode('utf-8', errors='replace')
            decoded = encoded.decode('utf-8', errors='replace')
            print(decoded)
        except:
            ascii_text = text.encode('ascii', errors='ignore').decode('ascii')
            print(f"{ascii_text} [Thai text removed due to encoding issues]")

def test_image_prediction():
    base_url = "http://localhost:8000"
    image_path = "test_images/1 (10).jpg"

    safe_print("=== Testing Image Prediction Endpoint ===")

    # Test health endpoint first
    safe_print("\n1. Testing /health...")
    try:
        response = requests.get(f"{base_url}/health")
        safe_print(f"Health Status: {response.status_code}")
        if response.status_code == 200:
            safe_print("✅ Model is loaded and ready")
        else:
            safe_print(f"❌ Health check failed: {response.text}")
    except Exception as e:
        safe_print(f"❌ Health check error: {e}")

    # Test image prediction
    safe_print("\n2. Testing /predict...")
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': ('test_image.jpg', image_file, 'image/jpeg')}
            response = requests.post(f"{base_url}/predict", files=files)

        safe_print(f"Predict Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            safe_print(f"Filename: {result.get('filename')}")
            safe_print(f"Model Mode: {result.get('model_mode')}")

            top_pred = result.get('top_prediction', {})
            safe_print(f"Top Prediction:")
            safe_print(f"  Code: {top_pred.get('code')}")
            safe_print(f"  Name: {top_pred.get('name')}")
            safe_print(f"  Confidence: {top_pred.get('probability_pct')}%")

            all_preds = result.get('all_predictions', [])
            safe_print(f"\nAll Predictions ({len(all_preds)} classes):")
            for i, pred in enumerate(all_preds[:5]):  # Show top 5
                safe_print(f"  {pred['rank']}. [{pred['code']}] {pred['name']}: {pred['probability_pct']}%")

            safe_print("\n✅ Image prediction successful!")
        else:
            safe_print(f"❌ Prediction failed: {response.text}")

    except FileNotFoundError:
        safe_print(f"❌ Test image not found: {image_path}")
    except Exception as e:
        safe_print(f"❌ Prediction error: {e}")

if __name__ == "__main__":
    test_image_prediction()