# Deriv AI Trading Bot

An advanced AI-powered trading bot for Deriv platform with smart martingale system and predictive analysis.

## Features

- 🤖 AI-powered trade analysis using OpenAI GPT
- 📊 Advanced technical analysis with multiple indicators
- 🎯 Smart Martingale system with risk management
- 📈 Real-time market data processing
- 💾 Historical data analysis and pattern recognition
- 🔄 Automated trade execution
- 📋 Comprehensive logging and analytics
- 🛡️ Risk management and stop-loss mechanisms

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```
DERIV_TOKEN=your_deriv_token
OPENAI_API_KEY=your_openai_key
```

3. Run the bot:
```bash
python main.py
```

## Configuration

Edit `config.py` to customize:
- Trading parameters
- Risk management settings
- AI analysis preferences
- Martingale system behavior

## Safety Features

- Maximum consecutive losses limit
- Daily loss limits
- Position size limits
- Emergency stop mechanisms

## Disclaimer

⚠️ **Trading involves significant risk. Use this bot at your own risk. Always test with demo accounts first.**
