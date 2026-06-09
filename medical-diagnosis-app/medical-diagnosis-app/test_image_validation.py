#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Image Validation Endpoint
"""
import requests
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

def test_image_validation():
    base_url = "http://localhost:8000"
    image_path = "test_images/1 (10).jpg"

    safe_print("=== Testing Image Validation Endpoint ===")

    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': ('test_image.jpg', image_file, 'image/jpeg')}
            response = requests.post(f"{base_url}/validate-image", files=files)

        safe_print(f"Validation Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            safe_print(f"Is Valid: {result.get('is_valid')}")
            safe_print(f"Confidence: {result.get('confidence')}%")
            safe_print(f"Message: {result.get('message')}")

            if result.get('is_valid'):
                safe_print("\n✅ Image validation successful! It's a dental X-ray.")
            else:
                safe_print("\n❌ Image validation failed! Not a dental X-ray.")
        else:
            safe_print(f"❌ Validation failed: {response.text}")

    except FileNotFoundError:
        safe_print(f"❌ Test image not found: {image_path}")
    except Exception as e:
        safe_print(f"❌ Validation error: {e}")

if __name__ == "__main__":
    test_image_validation()