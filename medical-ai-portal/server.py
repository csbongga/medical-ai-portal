#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Medical Portal Server
Simple HTTP server for the main portal page
"""
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 3000
DIRECTORY = Path(__file__).parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def log_message(self, format, *args):
        # Customize log format
        print(f" {self.address_string()} - {format % args}")

def start_server():
    """Start the AI Medical Portal server"""
    print(" Starting AI Medical Portal Server...")
    print(f" Server will be available at: http://localhost:{PORT}")
    print(f" Serving files from: {DIRECTORY}")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)

    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"SUCCESS: Server running at http://localhost:{PORT}")
            print(f" Opening browser...")

            # Open browser automatically
            webbrowser.open(f"http://localhost:{PORT}")

            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n Server stopped by user")
    except OSError as e:
        if e.errno == 10048:  # Port already in use
            print(f"ERROR: Port {PORT} is already in use!")
            print("Try closing other applications or use a different port")
        else:
            print(f"ERROR: Server failed to start: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")

if __name__ == "__main__":
    # Check if index.html exists
    index_file = DIRECTORY / "index.html"
    if not index_file.exists():
        print("ERROR: Error: index.html not found!")
        print(f"Please make sure index.html exists in: {DIRECTORY}")
        exit(1)

    start_server()