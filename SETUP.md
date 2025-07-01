# 🤖 Deriv AI Trading Bot - Installation Guide

## Prerequisites

1. **Python 3.8 or higher** - Download from [python.org](https://python.org)
2. **Deriv API Token** - Get from [Deriv API](https://app.deriv.com/account/api-token)
3. **OpenAI API Key** - Get from [OpenAI Platform](https://platform.openai.com/api-keys)

## Quick Setup (Windows)

1. **Clone or download** this project to your computer
2. **Double-click** `start_bot.bat` to automatically set up and start the bot
3. The script will:
   - Create a virtual environment
   - Install all dependencies
   - Set up the database
   - Start the trading bot

## Manual Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment

1. Copy `.env.example` to `.env`
2. Edit `.env` file with your credentials:

```env
DERIV_TOKEN=your_deriv_token_here
OPENAI_API_KEY=your_openai_api_key_here
DEMO_MODE=true
```

### 3. Initialize Database

```bash
python database.py
```

### 4. Start the Bot

```bash
# Start trading bot
python main.py

# Start web dashboard (in another terminal)
python dashboard.py
```

## Configuration

### Trading Settings

Edit `config.py` to customize:

- **Initial Stake**: Starting trade amount
- **Risk Management**: Daily loss limits, position sizes
- **Martingale System**: Progression parameters
- **AI Configuration**: Model settings, confidence thresholds

### Martingale System

The smart martingale system:
- Maintains same stake for first 3 losses
- Doubles stake after 3+ consecutive losses
- Includes unpredictability factors
- Has comprehensive risk management

### AI Analysis

The bot uses OpenAI GPT for:
- Market trend analysis
- Technical indicator interpretation
- Trading signal generation
- Risk assessment

## Web Dashboard

Access the dashboard at `http://localhost:8000` to monitor:
- Real-time trading statistics
- Performance charts
- Recent trades
- AI analysis results
- Risk management status

## Important Notes

⚠️ **Trading Risks**
- Start with demo mode (`DEMO_MODE=true`)
- Test thoroughly before live trading
- Never risk more than you can afford to lose
- Monitor the bot regularly

🔒 **Security**
- Keep your API tokens secure
- Don't share your `.env` file
- Use strong passwords
- Enable 2FA on your accounts

📊 **Performance**
- Monitor win rate and profitability
- Adjust parameters based on market conditions
- Review AI analysis quality
- Keep trading logs for analysis

## Troubleshooting

### Common Issues

1. **"Import error: No module named..."**
   - Run: `pip install -r requirements.txt`

2. **"Connection failed"**
   - Check your internet connection
   - Verify Deriv API token is valid
   - Ensure you're not blocked by firewall

3. **"Database error"**
   - Run: `python database.py` to recreate tables

4. **"AI analysis failed"**
   - Check OpenAI API key is valid
   - Verify you have API credits

### Getting Help

1. Check the logs in the `logs/` directory
2. Review error messages in the console
3. Verify your configuration in `.env`
4. Test with demo mode first

## Advanced Features

### Custom Strategies

You can modify the AI prompt in `ai_analyzer.py` to:
- Focus on specific indicators
- Adjust risk tolerance
- Target different market conditions

### Technical Indicators

The bot includes:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages
- Support/Resistance levels
- Candlestick patterns

### Risk Management

Built-in protections:
- Daily loss limits
- Position size limits
- Consecutive loss limits
- Volatility adjustments
- Emergency stop mechanisms

## Disclaimer

This trading bot is for educational and research purposes. Trading involves significant financial risk, and you should never trade with money you cannot afford to lose. The developers are not responsible for any financial losses incurred through the use of this software.

Always test strategies in demo mode before live trading, and consider consulting with a financial advisor before making investment decisions.
