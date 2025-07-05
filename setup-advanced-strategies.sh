#!/bin/bash

echo "🚀 Setting up Deriv Trading Bot with Advanced Strategies..."
echo ""

# Backend setup
echo "📦 Installing Python dependencies..."
cd backend
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Backend dependencies installed successfully!"
else
    echo "❌ Failed to install backend dependencies!"
    exit 1
fi

# Frontend setup
echo "📦 Installing Node.js dependencies..."
cd ../frontend
npm install

if [ $? -eq 0 ]; then
    echo "✅ Frontend dependencies installed successfully!"
else
    echo "❌ Failed to install frontend dependencies!"
    exit 1
fi

echo ""
echo "🎯 Setup complete! The bot now includes 15 advanced trading strategies:"
echo ""
echo "1. Adaptive Mean Reversion Rebound"
echo "2. Micro-Trend Momentum Tracker"
echo "3. RSI-Tick Divergence Detector"
echo "4. Volatility Spike Fader"
echo "5. Tick Flow Strength Pulse"
echo "6. Double Confirmation Breakout"
echo "7. RSI Overextension Fade"
echo "8. Multi-Tick Pivot Bounce"
echo "9. MACD-Momentum Sync Engine"
echo "10. Time-of-Tick Scalper"
echo "11. Volatility Collapse Compression"
echo "12. Two-Step Confirmation Model"
echo "13. Inverted Divergence Flip"
echo "14. Cumulative Strength Index Pullback"
echo "15. Tri-Indicator Confluence Strategy"
echo ""
echo "🧠 Features:"
echo "• Real-time strategy scanning"
echo "• Technical indicator analysis (RSI, MACD, Bollinger Bands, etc.)"
echo "• Confidence-based trade execution"
echo "• Strategy name display for each trade"
echo "• Live indicator dashboard"
echo ""
echo "To start the application:"
echo "1. Run: python backend/app.py"
echo "2. Run: npm start (in frontend directory)"
echo "3. Open http://localhost:8080"
