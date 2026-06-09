#!/usr/bin/env python3
"""
Manual Test Script for Claude API Integration
"""
import json
import urllib.request
import urllib.parse

def test_claude_status():
    print("=== Testing Claude Status ===")
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/claude-status")
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"Status: {response.code}")
            print(f"Claude Status: {result.get('status')}")
            print(f"Model: {result.get('model')}")
            print(f"Message: {result.get('message')}")
            return result.get('status') == 'success'
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_treatment_advice():
    print("\n=== Testing Treatment Advice ===")
    try:
        payload = {
            "disease_name": "Irreversible pulpitis with Acute periodontitis",
            "disease_code": "IP",
            "confidence": 89.5,
            "patient_info": "Patient age 35, severe toothache for 2 days"
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://127.0.0.1:8000/treatment-advice",
            data=data,
            method="POST"
        )
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"Status: {response.code}")
            print(f"Success: {result.get('success')}")
            print(f"Message: {result.get('message')}")

            recommendation = result.get('data', {}).get('recommendation')
            if recommendation:
                print(f"Recommendation: {recommendation[:300]}...")
                return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def test_chat():
    print("\n=== Testing Chat ===")
    try:
        payload = {
            "question": "What are the key considerations for this diagnosis?",
            "context": {
                "disease": "IP",
                "confidence": "89.5%"
            }
        }

        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            "http://127.0.0.1:8000/chat",
            data=data,
            method="POST"
        )
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"Status: {response.code}")
            print(f"Success: {result.get('success')}")

            chat_response = result.get('data', {}).get('response')
            if chat_response:
                print(f"Chat Response: {chat_response[:300]}...")
                return True

    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Manual Tests...")

    # Test 1: Claude Status
    status_ok = test_claude_status()

    if status_ok:
        # Test 2: Treatment Advice
        treatment_ok = test_treatment_advice()

        # Test 3: Chat
        chat_ok = test_chat()

        print(f"\n=== Test Results ===")
        print(f"Claude Status: {'✅ PASS' if status_ok else '❌ FAIL'}")
        print(f"Treatment Advice: {'✅ PASS' if treatment_ok else '❌ FAIL'}")
        print(f"Chat: {'✅ PASS' if chat_ok else '❌ FAIL'}")

        if all([status_ok, treatment_ok, chat_ok]):
            print("\n🎉 ALL TESTS PASSED! Claude API is working!")
        else:
            print("\n⚠️ Some tests failed. Check the errors above.")
    else:
        print("\n❌ Claude API connection failed. Check your API key.")