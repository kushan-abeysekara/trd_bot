#!/bin/bash

# AI Trading Bot Setup and Run Script

echo "🤖 AI Trading Bot - Setup and Launch"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16 or higher."
    exit 1
fi

echo "✅ Python and Node.js detected"

# Setup Backend
echo ""
echo "🔧 Setting up Backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Initialize database
echo "🗄️ Initializing database..."
python3 -c "
from app import create_app, db
app = create_app()
with app.app_context():
    db.create_all()
    print('Database initialized successfully!')
"

cd ..

# Setup Frontend
echo ""
echo "🎨 Setting up Frontend..."
cd frontend

# Install Node dependencies
echo "📥 Installing Node.js dependencies..."
npm install

cd ..

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Starting AI Trading Bot..."
echo ""

# Start backend in background
echo "🔙 Starting Backend server..."
cd backend
source venv/bin/activate
python3 app.py &
BACKEND_PID=$!

cd ..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🖥️ Starting Frontend server..."
cd frontend
npm start &
FRONTEND_PID=$!

cd ..

echo ""
echo "🎉 AI Trading Bot is now running!"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔗 API: http://localhost:5000"
echo ""
echo "Features:"
echo "• 🤖 AI-powered trading with ML models for all contract types"
echo "• 📈 Advanced money management with smart Martingale"
echo "• 🛡️ Safety limits (daily stop loss, daily target, cooldowns)"
echo "• 🔄 Dynamic strategy switching (3 modes)"
echo "• 📊 All 10 Deriv contract types supported"
echo "• 🧠 ChatGPT integration for auto-training"
echo "• 💾 Local ML training data saved in SQL database"
echo "• 📱 Real-time monitoring and control"
echo "• 🎯 Live Trading Bot with Start/Stop controls"
echo "• 📋 Real-time open trades display"
echo "• 📊 Complete trading history with P&L"
echo "• 💰 Auto stake management with manual override"
echo "• 🎨 Advanced Mean Reversion Strategy"
echo "• 🔍 Real-time strategy status display"
echo "• 🤖 Comprehensive Bot Control Panel"
echo "• 📊 Real-time bot performance monitoring"
echo "• 🎯 Active trades display with force close"
echo "• 📈 Live bot statistics and P&L tracking"
echo "• ⚡ Instant bot start/stop controls"
echo "• 🔄 Auto-refreshing bot status (5s intervals)"
echo "• 📱 Mobile-responsive bot interface"
echo "• 🎛️ Real-Time Strategy Monitor Window"
echo "• 📊 Live Technical Indicators Dashboard"
echo "• ✅ Strategy Condition Tracker (4/4 indicators)"
echo "• 🎯 Visual Signal Analysis with status checks"
echo "• ⚡ Live market data with condition validation"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo ''; echo '🛑 Shutting down...'; kill $BACKEND_PID $FRONTEND_PID; exit 0" INT
wait
