#!/usr/bin/env python3
"""
ทดสอบ Claude Chat API เพื่อตรวจหาปัญหา
"""
import requests
import json

# ทดสอบ claude status ก่อน
def test_claude_status():
    try:
        response = requests.get("http://localhost:8000/claude-status")
        print("🔍 Claude Status:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("✅ Result:", json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("❌ Error:", response.text)
        return response.status_code == 200
    except Exception as e:
        print("❌ Connection Error:", str(e))
        return False

# ทดสอบ chat endpoint
def test_chat():
    try:
        payload = {
            "question": "สวัสดีครับ คุณเป็นใครครับ?",
            "conversation_history": [],
            "context": {}
        }

        print("🔍 Sending chat request...")
        print("📤 Payload:", json.dumps(payload, ensure_ascii=False, indent=2))

        response = requests.post(
            "http://localhost:8000/chat",
            headers={"Content-Type": "application/json"},
            json=payload
        )

        print("🔍 Chat Response Status:", response.status_code)
        print("🔍 Raw Response:", response.text[:500] + "..." if len(response.text) > 500 else response.text)

        if response.status_code == 200:
            result = response.json()
            print("✅ Chat Result:")
            print("📥 Success:", result.get("success", False))
            print("📥 Message:", result.get("message", ""))
            if result.get("data"):
                print("📥 Response:", result["data"].get("response", ""))
        else:
            print("❌ Chat Error:", response.text)

    except Exception as e:
        print("❌ Chat Request Error:", str(e))

if __name__ == "__main__":
    print("🚀 Testing Chat Functionality...")
    print("=" * 50)

    print("\n1️⃣ Testing Claude Status...")
    status_ok = test_claude_status()

    print(f"\n2️⃣ Testing Chat Endpoint...")
    if status_ok:
        test_chat()
    else:
        print("⚠️ Claude status failed, skipping chat test")

    print("\n✅ Test completed!")