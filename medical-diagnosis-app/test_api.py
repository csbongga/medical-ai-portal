#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Claude API Endpoints
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

def test_claude_endpoints():
    base_url = "http://localhost:8000"

    safe_print("=== Testing Claude API Endpoints ===")

    # 1. Test Claude status
    safe_print("\n1. Testing /claude-status...")
    try:
        response = requests.get(f"{base_url}/claude-status")
        safe_print(f"Status: {response.status_code}")
        safe_print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        safe_print(f"Error: {e}")

    # 2. Test treatment advice
    safe_print("\n2. Testing /treatment-advice...")
    try:
        payload = {
            "disease_name": "Irreversible pulpitis with Acute periodontitis",
            "disease_code": "IP",
            "confidence": 89.5,
            "patient_info": "ผู้ป่วยอายุ 35 ปี มีอาการปวดฟันรุนแรงเป็นระยะเวลา 2 วัน"
        }
        response = requests.post(f"{base_url}/treatment-advice", json=payload)
        safe_print(f"Status: {response.status_code}")
        result = response.json()
        safe_print(f"Success: {result.get('success')}")
        safe_print(f"Message: {result.get('message')}")
        if result.get('data', {}).get('recommendation'):
            recommendation = result['data']['recommendation'][:200] + "..."
            safe_print(f"Recommendation: {recommendation}")

            # Test JSON parsing
            try:
                full_recommendation = result['data']['recommendation']
                parsed = json.loads(full_recommendation)
                safe_print(f"✅ JSON parsing successful! Contains {len(parsed.get('treatments', []))} treatments")
            except json.JSONDecodeError as e:
                safe_print(f"❌ JSON parsing failed: {e}")

    except Exception as e:
        safe_print(f"Error: {e}")

    # 3. Test chat
    safe_print("\n3. Testing /chat...")
    try:
        payload = {
            "question": "ในกรณีที่ผู้ป่วยมีแพ้ยาปฏิชีวนะ มีตัวเลือกการรักษาอื่นไหม?",
            "context": {
                "disease": "IP",
                "confidence": "89.5%"
            }
        }
        response = requests.post(f"{base_url}/chat", json=payload)
        safe_print(f"Status: {response.status_code}")
        result = response.json()
        safe_print(f"Success: {result.get('success')}")
        safe_print(f"Message: {result.get('message')}")
        if result.get('data', {}).get('response'):
            chat_response = result['data']['response'][:200] + "..."
            safe_print(f"Chat Response: {chat_response}")
    except Exception as e:
        safe_print(f"Error: {e}")

if __name__ == "__main__":
    test_claude_endpoints()