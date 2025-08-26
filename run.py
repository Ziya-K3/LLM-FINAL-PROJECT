#!/usr/bin/env python3
"""
NeuroAssist - Brain MRI Analysis System
Startup script for the FastAPI application
"""

import uvicorn
import os
import sys

def main():
    """Start the NeuroAssist application"""
    
    # Check if required directories exist
    required_dirs = ['uploads', 'reports', 'neuroassist_analyses']
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("🧠 NeuroAssist - Brain MRI Analysis System")
    print("=" * 50)
    print("Starting server...")
    print("Access the application at: http://localhost:8080")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1) 