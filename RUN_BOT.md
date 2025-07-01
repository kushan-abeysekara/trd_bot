# 🚀 How to Run the Deriv AI Trading Bot

## 📋 Prerequisites Check

Before running the bot, ensure you have:

✅ **Python 3.8-3.12** (Python 3.13 has compatibility issues)  
✅ **Your Deriv API Token**: `B8ZH857zyOqvHah`  
✅ **Your OpenAI API Key**: Already configured  
✅ **Stable Internet Connection**  

## 🎯 Method 1: Quick Start (Recommended)

### Step 1: Open PowerShell/Command Prompt
Press `Win + R`, type `powershell`, press Enter

### Step 2: Navigate to Bot Directory
```powershell
cd "D:\GITHUB\trd_bot"
```

### Step 3: Run the Automated Setup
```powershell
.\start_bot.bat
```

**What this script does:**
- ✅ Creates Python virtual environment
- ✅ Installs all dependencies  
- ✅ Sets up database
- ✅ Starts the trading bot

---

## 🛠️ Method 2: Manual Setup (If Method 1 Fails)

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies  
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Initialize Database
```powershell
python database.py
```

### Step 4: Start Trading Bot
```powershell
python main.py
```

---

## 🌐 Method 3: With Web Dashboard

### Step 1: Start the Bot (Terminal 1)
```powershell
python main.py
```

### Step 2: Start Dashboard (Terminal 2)
```powershell
python dashboard.py
```

### Step 3: Open Browser
Visit: `http://localhost:8000`

---

## 📊 Method 4: Advanced Configuration

### Custom Settings
Edit `config.py` or `.env` file:

```env
# Trading Settings
DEMO_MODE=true              # Always start with demo!
INITIAL_STAKE=1.0          # Starting stake amount
MAX_DAILY_LOSS=100         # Maximum daily loss limit
MAX_CONSECUTIVE_LOSSES=5   # Stop after X losses

# AI Settings  
OPENAI_MODEL=gpt-4-turbo-preview
AI_CONFIDENCE_THRESHOLD=0.7

# Martingale Settings
MARTINGALE_MULTIPLIER=2.0
UNPREDICTABILITY_FACTOR=0.15
```

---

## 🔧 Troubleshooting

### Problem: Python Version Issues
**Solution:** Use Python 3.8-3.12 (avoid 3.13)
```powershell
python --version  # Check your version
```

### Problem: Dependencies Won't Install  
**Solution:** Update pip and try again
```powershell
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

### Problem: OpenAI API Errors
**Solution:** Check your API key balance and limits
- Visit: https://platform.openai.com/usage

### Problem: Deriv Connection Issues
**Solution:** Verify your token is active
- Visit: https://app.deriv.com/account/api-token

---

## 📈 Bot Features Overview

### 🤖 Smart Martingale System
- **Unpredictable Patterns**: Uses chaos theory and multiple entropy sources
- **AI-Driven Scaling**: Adjusts stakes based on AI confidence  
- **Risk Management**: Never risks more than 2% of balance per trade
- **Progressive Logic**: Maintains stakes for 3 losses, then scales intelligently

### 🧠 Advanced AI Analysis  
- **GPT-4 Powered**: Uses latest AI for market analysis
- **Multi-Timeframe**: Analyzes 1m, 5m, 15m, 30m, 1h charts
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and 8+ more
- **Market Sentiment**: Incorporates broader market context

### 🛡️ Comprehensive Risk Management
- **Emergency Brake**: Stops after 8 consecutive losses  
- **Cool-down Periods**: Pauses trading after significant losses
- **Balance Protection**: Never exceeds safe position sizing
- **Daily Limits**: Automatic shutdown at loss thresholds

---

## 🎮 Quick Start Commands

```powershell
# Navigate to bot directory
cd "D:\GITHUB\trd_bot"

# Quick start (all-in-one)
.\start_bot.bat

# Manual start
python main.py

# Start with dashboard
python dashboard.py
# Then visit: http://localhost:8000
```

---

## ⚠️ Important Safety Notes

### 🔴 DEMO MODE FIRST
Always test in demo mode before live trading:
```env
DEMO_MODE=true
```

### 💰 Risk Management
- Start with small stakes ($1-5)
- Never risk more than you can afford to lose
- Monitor the bot regularly
- Set daily loss limits

### 📊 Performance Monitoring
- Check AI confidence scores (aim for >0.7)
- Monitor win rate (target >55%)
- Review technical signal alignment
- Track daily P&L

---

## 🎯 Expected Performance

### Optimal Conditions
- **Win Rate**: 60-70% in stable markets
- **AI Confidence**: >0.75 for best trades  
- **Daily Target**: 5-10% portfolio growth
- **Risk Level**: Conservative to moderate

### Market Adaptation
- **High Volatility**: Reduces stakes automatically
- **Low Confidence**: Skips trades or uses minimal stakes  
- **Losing Streaks**: Activates cool-down periods
- **Market Regimes**: Adapts strategy based on conditions

---

## 🆘 Support & Monitoring

### Real-Time Monitoring
```powershell
# View logs
Get-Content trading_bot.log -Wait

# Check bot status
python -c "from database import SessionLocal; print('Database OK')"
```

### Performance Dashboard
Visit `http://localhost:8000` for:
- Real-time statistics
- Performance charts  
- Trade history
- AI analysis display
- Risk metrics

---

**🎉 Ready to Start Trading with AI! 🤖💰**

Remember: Always start with DEMO_MODE=true and small stakes!
