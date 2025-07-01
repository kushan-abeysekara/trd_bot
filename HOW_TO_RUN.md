# 🚀 How to Run the Deriv AI Trading Bot

## Quick Start (Recommended)

### Option 1: Use the Easy Setup Script (Windows)
1. **Double-click** `easy_setup.bat`
2. Follow the prompts
3. The script will automatically:
   - Install all required packages
   - Create configuration files
   - Let you choose what to run

### Option 2: Use Python Setup Script  
1. Open Command Prompt/PowerShell in the project folder
2. Run: `python install_and_run.py`
3. Follow the interactive prompts

## Manual Installation

### 1. Install Python Dependencies
```bash
# Install with the updated requirements
pip install -r requirements.txt

# If you get numpy errors, try:
pip install "numpy>=1.21.0,<1.27.0" --no-build-isolation
pip install "pandas>=2.0.0" --no-build-isolation
```

### 2. Setup Environment
Create a `.env` file with your API credentials:
```env
# Environment Configuration
DERIV_TOKEN=your_deriv_token_here
OPENAI_API_KEY=your_openai_key_here

# Database Configuration  
DATABASE_URL=sqlite:///trading_bot.db

# Trading Configuration
DEMO_MODE=true
MAX_DAILY_LOSS=100
MAX_CONSECUTIVE_LOSSES=5  
INITIAL_STAKE=1.0

# WebSocket Configuration
DERIV_WS_URL=wss://ws.binaryws.com/websockets/v3?app_id=1089
```

### 3. Create Required Directories
```bash
mkdir logs
mkdir static  
mkdir templates
```

## Running the Application

### Option A: Trading Bot Only
```bash
python main.py
```

### Option B: Web Dashboard Only  
```bash
python dashboard.py
```
Then open: http://localhost:8000

### Option C: Both (Recommended)
1. **Terminal 1**: `python dashboard.py`
2. **Terminal 2**: `python main.py`
3. Open browser: http://localhost:8000

## Getting API Keys

### Deriv API Token
1. Go to [Deriv.com](https://deriv.com)
2. Login to your account
3. Go to Settings → API Token
4. Create a new token with trading permissions
5. Copy the token to your `.env` file

### OpenAI API Key
1. Go to [OpenAI](https://platform.openai.com)
2. Create an account or login
3. Go to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## Troubleshooting

### Common Issues

#### 1. "No matching distribution found for numpy"
**Solution**: 
```bash
pip install "numpy>=1.21.0,<1.27.0" --no-build-isolation
```

#### 2. "Microsoft Visual C++ required"
**Solution**: Install Microsoft Visual C++ Build Tools or use:
```bash
pip install --only-binary=all numpy pandas scikit-learn
```

#### 3. Python 3.13 Compatibility Issues
**Solution**: Use Python 3.8-3.12 instead:
- Download from [python.org](https://python.org)
- Choose version 3.11 or 3.12 for best compatibility

#### 4. "Module not found" errors  
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### 5. Permission errors on Windows
**Solution**: Run Command Prompt as Administrator

### Package Installation Alternatives

If the main requirements fail, try installing packages individually:

```bash
# Core packages
pip install websockets requests python-dotenv fastapi uvicorn
pip install sqlalchemy aiofiles jinja2 python-multipart httpx
pip install pymysql schedule openai

# Scientific packages  
pip install numpy --no-build-isolation
pip install pandas --no-build-isolation  
pip install scikit-learn

# Optional packages
pip install matplotlib seaborn cryptography psutil
```

## Usage Guide

### 1. First Run
- Bot starts in **DEMO MODE** by default (safe)
- Initial stake is $1.00
- Maximum daily loss is $100

### 2. Web Dashboard Features
- Real-time trading statistics
- Live trade monitoring  
- Performance charts
- Start/Stop bot controls
- AI analysis results

### 3. Safety Features
- Demo mode protection
- Maximum loss limits
- Emergency brake system
- Consecutive loss protection

### 4. Configuration
Edit `.env` file to customize:
- Trading parameters
- Risk limits  
- API endpoints
- Database settings

## File Structure
```
trd_bot/
├── main.py              # Main trading bot
├── dashboard.py         # Web dashboard
├── deriv_api.py         # Deriv API connection
├── technical_analysis.py # Technical indicators
├── ai_analyzer.py       # AI analysis
├── martingale_system.py # Risk management
├── database.py          # Database models
├── config.py            # Configuration
├── requirements.txt     # Dependencies
├── .env                 # Environment variables
├── easy_setup.bat       # Windows setup script
├── install_and_run.py   # Python setup script
└── logs/                # Log files
```

## Support

If you encounter issues:
1. Check the `logs/` directory for error details
2. Ensure all API keys are correctly set
3. Verify Python version (3.8-3.12 recommended)
4. Try the automatic setup scripts first

## Important Notes

⚠️ **SAFETY FIRST**
- Always test in DEMO mode first
- Never risk more than you can afford to lose
- Monitor the bot regularly
- Start with small stakes

✅ **RECOMMENDED WORKFLOW**
1. Run `easy_setup.bat` (Windows) or `install_and_run.py`
2. Update `.env` with your API keys
3. Start dashboard: `python dashboard.py`
4. Open http://localhost:8000 in browser
5. Use dashboard to start/stop the trading bot
6. Monitor performance through the web interface
