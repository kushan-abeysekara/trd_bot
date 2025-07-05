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

from app import app, socketio
from config import FLASK_HOST, FLASK_PORT

if __name__ == '__main__':
    # Production mode with Gunicorn-compatible settings
    # Increase ping interval and timeout for more reliable WebSocket connections
    socketio.server.eio.ping_interval = 25
    socketio.server.eio.ping_timeout = 60
    
    socketio.run(
        app,
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        allow_unsafe_werkzeug=True,
        use_reloader=False,
        log_output=True,  # Enable logging
        cors_allowed_origins='*'  # Temporarily allow all origins for testing
    )
