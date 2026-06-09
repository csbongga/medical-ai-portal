#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Claude API Connection
"""
import sys
import os

# Fix Windows console encoding for Thai text
if sys.platform.startswith('win'):
    try:
        # Set console to UTF-8
        os.system('chcp 65001 > nul')
        # Reconfigure stdout for UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def safe_print(text, fallback_encoding='ascii'):
    """Print text with proper encoding handling"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: encode to bytes then decode with error handling
        try:
            encoded = text.encode('utf-8', errors='replace')
            decoded = encoded.decode('utf-8', errors='replace')
            print(decoded)
        except:
            # Final fallback: remove non-ASCII characters
            ascii_text = text.encode('ascii', errors='ignore').decode('ascii')
            print(f"{ascii_text} [Thai text removed due to encoding issues]")

if __name__ == "__main__":
    try:
        from claude_service import claude_service

        safe_print("=== Testing Claude API Connection ===")
        result = claude_service.test_connection()
        safe_print(f"Status: {result['status']}")
        safe_print(f"Message: {result.get('message', 'No message')}")

        if result['status'] == 'success':
            safe_print(f"Model: {result['model']}")
            safe_print(f"Response: {result.get('response', 'No response')}")
            safe_print("\nSUCCESS: Claude API is ready!")
        else:
            safe_print("\nERROR: Connection failed")

    except Exception as e:
        safe_print(f"ERROR: {e}")