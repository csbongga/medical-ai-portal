#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Claude API JSON Response
"""
import sys
import os
import json

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

if __name__ == "__main__":
    try:
        from claude_service import claude_service

        safe_print("=== Testing Claude API Treatment Recommendation (JSON Format) ===")

        # Test treatment recommendation
        result = claude_service.generate_treatment_recommendation(
            disease_name='Irreversible pulpitis with Acute periodontitis',
            disease_code='IP',
            confidence=89.5,
            patient_info='ผู้ป่วยอายุ 35 ปี มีอาการปวดฟัน'
        )

        safe_print(f"Result type: {type(result)}")
        safe_print(f"First 200 characters: {result[:200]}")

        try:
            parsed = json.loads(result)
            safe_print("✅ JSON parsing successful!")
            safe_print(f"Location: {parsed.get('location', 'N/A')}")
            causes = parsed.get('causes', 'N/A')
            safe_print(f"Causes: {causes[:100]}...")
            treatments = parsed.get('treatments', [])
            safe_print(f"Treatments count: {len(treatments)}")

            if treatments:
                safe_print("\nFirst treatment:")
                safe_print(f"  Type: {treatments[0].get('type')}")
                safe_print(f"  Label: {treatments[0].get('label')}")
                safe_print(f"  Description: {treatments[0].get('description', '')[:80]}...")

        except json.JSONDecodeError as e:
            safe_print(f"❌ JSON parsing failed: {e}")
            safe_print("Raw response for debugging:")
            safe_print(result)

    except Exception as e:
        safe_print(f"ERROR: {e}")