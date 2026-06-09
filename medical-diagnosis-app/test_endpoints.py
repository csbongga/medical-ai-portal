#!/usr/bin/env python3
import json
import urllib.request
import urllib.parse

def test_endpoint(url, method="GET", data=None):
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, data=data, method=method)
            req.add_header('Content-Type', 'application/json')
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return response.code, result
    except Exception as e:
        return None, str(e)

def main():
    base_url = "http://127.0.0.1:8000"

    print("=== Testing Claude API Endpoints ===")

    # 1. Test Claude status
    print("\n1. Testing /claude-status...")
    status, result = test_endpoint(f"{base_url}/claude-status")
    print(f"Status: {status}")
    if result:
        print(f"Response: {json.dumps(result, ensure_ascii=True, indent=2)}")

    # 2. Test treatment advice
    print("\n2. Testing /treatment-advice...")
    payload = {
        "disease_name": "Irreversible pulpitis with Acute periodontitis",
        "disease_code": "IP",
        "confidence": 89.5,
        "patient_info": "Patient age 35, severe toothache for 2 days"
    }
    status, result = test_endpoint(f"{base_url}/treatment-advice", "POST", payload)
    print(f"Status: {status}")
    if result and isinstance(result, dict):
        print(f"Success: {result.get('success')}")
        print(f"Message: {result.get('message')}")
        if result.get('data', {}).get('recommendation'):
            print(f"Recommendation preview: {result['data']['recommendation'][:200]}...")

    # 3. Test chat
    print("\n3. Testing /chat...")
    payload = {
        "question": "What are alternative treatments if patient is allergic to antibiotics?",
        "context": {
            "disease": "IP",
            "confidence": "89.5%"
        }
    }
    status, result = test_endpoint(f"{base_url}/chat", "POST", payload)
    print(f"Status: {status}")
    if result and isinstance(result, dict):
        print(f"Success: {result.get('success')}")
        print(f"Message: {result.get('message')}")
        if result.get('data', {}).get('response'):
            print(f"Chat Response preview: {result['data']['response'][:200]}...")

if __name__ == "__main__":
    main()