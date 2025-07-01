#!/bin/bash

echo "Starting Deriv AI Trading Bot..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not available!"
    echo "Please install pip3"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment!"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies!"
    echo "Please check your internet connection and try again."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Initialize database
echo "Initializing database..."
python database.py

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo
    echo "WARNING: .env file not found!"
    echo "Please create a .env file with your configuration."
    echo "See .env.example for reference."
    echo
fi

# Start the bot
echo
echo "Starting the trading bot..."
echo "Press Ctrl+C to stop the bot"
echo
python main.py
