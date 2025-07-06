#!/usr/bin/env python3
"""
Production server for Deriv Trading Bot
Use this script to run the application in production with Gunicorn
"""
import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app import app
from config import FLASK_HOST, FLASK_PORT

if __name__ == '__main__':
    # Production mode with Gunicorn-compatible settings
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        threaded=True,  # Use threading for better performance
        use_reloader=False
    )
