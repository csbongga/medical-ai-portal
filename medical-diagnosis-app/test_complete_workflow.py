#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Complete Application Workflow
This test simulates the full user experience:
1. Upload image
2. Get prediction
3. Generate treatment advice
4. Chat with Claude
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

def test_complete_workflow():
    base_url = "http://localhost:8000"
    image_path = "test_images/1 (10).jpg"

    safe_print("=== Complete Application Workflow Test ===")

    # Step 1: Upload image and get prediction
    safe_print("\n🔬 Step 1: Image Prediction")
    safe_print("-" * 50)

    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': ('dental_xray.jpg', image_file, 'image/jpeg')}
            response = requests.post(f"{base_url}/predict", files=files)

        if response.status_code != 200:
            safe_print(f"❌ Prediction failed: {response.text}")
            return

        prediction_result = response.json()
        top_pred = prediction_result.get('top_prediction', {})

        safe_print(f"✅ Prediction successful!")
        safe_print(f"   Disease: {top_pred.get('name')}")
        safe_print(f"   Code: {top_pred.get('code')}")
        safe_print(f"   Confidence: {top_pred.get('probability_pct')}%")

    except Exception as e:
        safe_print(f"❌ Prediction error: {e}")
        return

    # Step 2: Generate treatment advice using Claude
    safe_print("\n💊 Step 2: Treatment Advice Generation")
    safe_print("-" * 50)

    try:
        treatment_payload = {
            "disease_name": top_pred.get('name'),
            "disease_code": top_pred.get('code'),
            "confidence": top_pred.get('probability_pct'),
            "patient_info": "ผู้ป่วยชาย อายุ 45 ปี มาด้วยอาการปวดฟันรุนแรง"
        }

        response = requests.post(f"{base_url}/treatment-advice", json=treatment_payload)

        if response.status_code != 200:
            safe_print(f"❌ Treatment advice failed: {response.text}")
            return

        treatment_result = response.json()

        if treatment_result.get('success'):
            safe_print("✅ Treatment advice generated!")
            recommendation = treatment_result['data']['recommendation']

            # Test if it's valid JSON
            try:
                parsed_recommendation = json.loads(recommendation)
                safe_print(f"   Location: {parsed_recommendation.get('location', 'N/A')[:80]}...")
                safe_print(f"   Causes: {parsed_recommendation.get('causes', 'N/A')[:100]}...")
                safe_print(f"   Treatments: {len(parsed_recommendation.get('treatments', []))} recommendations")

                # Show first treatment
                treatments = parsed_recommendation.get('treatments', [])
                if treatments:
                    first_treatment = treatments[0]
                    safe_print(f"   First treatment: [{first_treatment.get('type')}] {first_treatment.get('label', 'N/A')}")

            except json.JSONDecodeError as e:
                safe_print(f"⚠️ Treatment advice not in expected JSON format: {e}")
                safe_print(f"   Raw advice (first 200 chars): {recommendation[:200]}...")

        else:
            safe_print(f"❌ Treatment advice failed: {treatment_result.get('message', 'Unknown error')}")
            return

    except Exception as e:
        safe_print(f"❌ Treatment advice error: {e}")
        return

    # Step 3: Chat with Claude
    safe_print("\n💬 Step 3: Medical Chat")
    safe_print("-" * 50)

    try:
        chat_payload = {
            "question": "หากผู้ป่วยมีประวัติแพ้ยาปฏิชีวนะ ควรรักษาอย่างไร?",
            "context": {
                "disease": top_pred.get('name'),
                "confidence": f"{top_pred.get('probability_pct')}%",
                "patient_info": "ผู้ป่วยชาย อายุ 45 ปี มีประวัติแพ้ยาปฏิชีวนะ"
            }
        }

        response = requests.post(f"{base_url}/chat", json=chat_payload)

        if response.status_code != 200:
            safe_print(f"❌ Chat failed: {response.text}")
            return

        chat_result = response.json()

        if chat_result.get('success'):
            safe_print("✅ Chat response received!")
            chat_response = chat_result['data']['response']
            safe_print(f"   Response (first 200 chars): {chat_response[:200]}...")

        else:
            safe_print(f"❌ Chat failed: {chat_result.get('message', 'Unknown error')}")
            return

    except Exception as e:
        safe_print(f"❌ Chat error: {e}")
        return

    # Summary
    safe_print("\n🎉 Workflow Summary")
    safe_print("=" * 50)
    safe_print("✅ Image prediction: Working")
    safe_print("✅ Treatment advice: Working")
    safe_print("✅ Medical chat: Working")
    safe_print("✅ Complete workflow: SUCCESSFUL")
    safe_print("\nThe medical diagnosis application is fully functional!")

if __name__ == "__main__":
    test_complete_workflow()