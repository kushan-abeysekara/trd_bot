#!/bin/bash

echo "Starting Deriv AI Trading Bot Dashboard..."
echo

# Activate virtual environment
source venv/bin/activate

# Start the dashboard
echo "Starting web dashboard on http://localhost:8000"
echo "Press Ctrl+C to stop the dashboard"
echo
python dashboard.py
