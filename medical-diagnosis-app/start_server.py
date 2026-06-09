#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Start the Medical Diagnosis Application Server

This script starts the FastAPI server with proper configuration.
Make sure you're in the virtual environment before running this.

Usage:
    python start_server.py

Or manually with uvicorn:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import subprocess
import sys
import os

def start_server():
    """Start the FastAPI server using uvicorn"""
    print("🏥 Starting Medical Diagnosis Application Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")
    print("📱 Main interface at: http://localhost:8000")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)

    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the virtual environment:")
        print("   .\\venv\\Scripts\\Activate.ps1")
        print("2. Install dependencies if needed:")
        print("   pip install -r requirements.txt")
        print("3. Check if port 8000 is already in use")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found!")
        print("Please run this script from the medical-diagnosis-app directory")
        sys.exit(1)

    # Check if virtual environment is active (optional check)
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment may not be active")
        print("Consider running: .\\venv\\Scripts\\Activate.ps1")
        print("Continuing anyway...")

    start_server()