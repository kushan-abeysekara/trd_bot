# AI Trading Bot - Comprehensive Documentation

## 🤖 Overview

This is a comprehensive AI-powered trading bot for Deriv contracts that automatically selects the best trading strategies and manages risk using advanced machine learning algorithms.

## ✨ Key Features

### 🧠 AI & Machine Learning
- **Contract-Specific ML Models**: Separate trained models for each of the 10 contract types
- **Feature Extraction**: Advanced technical indicators and market microstructure analysis
- **ChatGPT Integration**: Auto-training and signal enhancement using OpenAI's GPT
- **Ensemble Learning**: Multiple ML algorithms (Random Forest, Gradient Boosting, SVM, Logistic Regression)
- **Real-time Learning**: Continuous model improvement with each trade

### 💰 Advanced Money Management
- **Smart Martingale**: Intelligent stake progression with volatility adjustments
- **Dynamic Stake Calculation**: Account balance, volatility, and performance-based sizing
- **Risk Protection**: Never risk more than 2% of account per trade
- **Consecutive Loss Protection**: Automatic stake reduction after losses

### 🛡️ Safety Limits
- **Daily Stop Loss**: Configurable percentage (default 10%)
- **Daily Target**: Automatic stop when target reached (default 20%)
- **Win Streak Cooldown**: Force pause after 2 wins + 1 loss for 3 signals
- **Account Protection**: Multiple layers of balance protection

### 🔄 Multi-Strategy System
- **Mode A: MA-RSI Trend Bot**: Moving average crossover with RSI filter
- **Mode B: Price Action Bounce**: Reversal strategy for sideways markets
- **Mode C: Random Entry Smart Exit**: Random entry with intelligent risk management

### 📊 All Deriv Contract Types
1. **Rise/Fall**: Basic binary prediction
2. **Touch/No Touch**: Barrier level prediction with volatility analysis
3. **In/Out (Boundary)**: Range-bound market prediction
4. **Asian Options**: Average price comparison with trend analysis
5. **Digits**: Last digit prediction using frequency analysis
6. **Reset Call/Put**: Barrier reset functionality with trend confirmation
7. **High/Low Ticks**: Short-term tick analysis
8. **Only Ups/Downs**: Directional consistency prediction
9. **Multipliers**: Leveraged trading with dynamic risk management
10. **Accumulators**: Growth-based contracts with knock-out protection

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- Deriv API token (Demo or Real account)

### Windows Installation
1. Download the project
2. Run `setup_and_run.bat`
3. Open http://localhost:3000
4. Set up your Deriv API token
5. Configure bot settings
6. Start trading!

### Linux/Mac Installation
1. Download the project
2. Run `chmod +x setup_and_run.sh && ./setup_and_run.sh`
3. Open http://localhost:3000
4. Set up your Deriv API token
5. Configure bot settings
6. Start trading!

## 🔧 Configuration

### Bot Settings
```javascript
{
  daily_stop_loss_percent: 10.0,    // Daily loss limit
  daily_target_percent: 20.0,       // Daily profit target  
  base_stake_percent: 2.0,          // Base stake per trade
  max_stake_percent: 10.0,          // Maximum stake limit
  cool_down_after_loss: 3,          // Signals to skip after losses
  strategy_switch_wins: 3,          // Wins before switching to Mode C
  strategy_switch_losses: 2,        // Losses before switching to Mode B
  reevaluate_trades: 10,            // Trades before strategy evaluation
  enable_martingale: true,          // Enable smart Martingale
  martingale_multiplier: 1.5,       // Martingale progression factor
  max_martingale_steps: 3           // Maximum Martingale steps
}
```

### Strategy Switching Rules
- **After 3 consecutive wins** → Switch to Mode C for 2 trades
- **After 2 losses** → Switch to Mode B for 3 trades
- **Every 10 trades** → Re-evaluate best-performing mode and continue

## 🎯 Contract-Specific Strategies

### Rise/Fall Contracts
- **ML Features**: Price momentum, technical indicators, trend analysis
- **Strategy**: Trend-following with RSI confirmation
- **Optimal Conditions**: Clear trending markets

### Touch/No Touch Contracts
- **ML Features**: Volatility analysis, barrier distance calculation
- **Strategy**: Volatility-based barrier prediction
- **Optimal Conditions**: Medium to high volatility markets

### Digit Contracts
- **ML Features**: Digit frequency analysis, price pattern recognition
- **Strategy**: Mean reversion on digit frequencies
- **Optimal Conditions**: High-frequency trading periods

### Multipliers
- **ML Features**: Risk-adjusted returns, downside protection
- **Strategy**: Trend following with tight risk management
- **Optimal Conditions**: Strong trending markets with low volatility

## 📈 Performance Monitoring

### Real-time Metrics
- Account balance and daily P&L
- Win rate and trade statistics
- Strategy performance comparison
- ML model accuracy per contract type
- Current market conditions

### Historical Analysis
- Trade history with detailed breakdowns
- Strategy performance over time
- ML model training progress
- Market condition correlations

## 🧠 Machine Learning Pipeline

### Data Collection
- Real-time market data ingestion
- Technical indicator calculation
- Price history and volume analysis
- Trade outcome tracking

### Feature Engineering
- **Price Features**: Returns, volatility, momentum
- **Technical Indicators**: RSI, MACD, Bollinger Bands, moving averages
- **Market Microstructure**: Order flow, tick patterns
- **Contract-Specific**: Digit frequencies, barrier distances, volatility clusters

### Model Training
- **Frequency**: Daily retraining for active contracts
- **Data**: Last 1000 trades per contract type
- **Validation**: 80/20 train/test split with time-series validation
- **Ensemble**: Best model selection based on accuracy

### ChatGPT Integration
- Market condition analysis and signal enhancement
- Automatic model improvement suggestions
- Performance review and optimization recommendations

## 🔒 Security & Risk Management

### Account Protection
- Multiple safety limits prevent significant losses
- Real-time balance monitoring
- Automatic shutdown on risk threshold breach

### API Security
- Secure token storage and validation
- Rate limiting and error handling
- Connection retry mechanisms

### Data Privacy
- Local database storage only
- No sensitive data transmission
- User-controlled API access

## 🛠️ Technical Architecture

### Backend (Python/Flask)
```
backend/
├── app.py                 # Main application
├── services/
│   ├── trading_bot_engine.py     # Core bot logic
│   ├── ml_strategies.py          # ML models and strategies
│   └── market_analyzer.py        # Real-time analysis
├── routes/
│   ├── trading_bot.py           # Bot control API
│   ├── ai_analysis.py           # AI analysis endpoints
│   └── deriv_api.py             # Deriv integration
└── models/
    └── user.py                  # Database models
```

### Frontend (React)
```
frontend/src/
├── components/
│   ├── AITradingBot.js         # Main bot interface
│   ├── AIMarketAnalyzer.js     # Market analysis
│   └── VolatilityChart.js      # Real-time charts
├── pages/
│   └── Dashboard.js            # Main dashboard
└── services/
    └── api.js                  # API integration
```

### Database Schema
```sql
-- Trading sessions
trading_sessions (id, user_id, start_time, end_time, pnl, trades, win_rate)

-- Individual trades
trade_results (id, user_id, contract_type, entry_price, exit_price, 
               stake, profit_loss, success, timestamp)

-- ML training data
ml_training_data (id, contract_type, market_data, indicators, 
                  outcome, timestamp)

-- Strategy performance
strategy_performance (id, user_id, strategy_name, contract_type, 
                      total_trades, wins, profit, win_rate)
```

## 📊 API Endpoints

### Bot Control
- `POST /api/trading-bot/start` - Start the trading bot
- `POST /api/trading-bot/stop` - Stop the trading bot
- `GET /api/trading-bot/status` - Get bot status
- `PUT /api/trading-bot/settings` - Update bot settings

### Analytics
- `GET /api/trading-bot/history` - Get trade history
- `GET /api/trading-bot/performance` - Get performance analytics
- `GET /api/trading-bot/sessions` - Get trading sessions

### ML Management
- `POST /api/trading-bot/ml-models/retrain` - Retrain ML models
- `GET /api/trading-bot/ml-models/performance` - Get ML performance
- `POST /api/trading-bot/test-signal` - Test signal generation

## 🎛️ Advanced Configuration

### OpenAI Integration
Set your OpenAI API key in the environment:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Deriv API Configuration
```bash
export DERIV_APP_ID="your-deriv-app-id"
export DERIV_WS_URL="wss://ws.binaryws.com/websockets/v3"
```

## 🐛 Troubleshooting

### Common Issues

1. **Bot won't start**
   - Check Deriv API token validity
   - Verify account has sufficient balance
   - Check internet connection

2. **Low win rate**
   - Allow more time for ML model training
   - Adjust risk settings
   - Check market conditions

3. **Connection errors**
   - Verify Deriv API endpoints
   - Check firewall settings
   - Restart the application

### Logs and Debugging
- Backend logs: `backend/logs/`
- Trading history: Available in dashboard
- Database: SQLite file `trading_bot.db`

## 📈 Performance Optimization

### ML Model Tuning
- Increase training data collection period
- Adjust feature engineering parameters
- Fine-tune model hyperparameters

### Risk Management
- Customize safety limits based on account size
- Adjust Martingale settings for risk tolerance
- Monitor strategy switching effectiveness

## 🔄 Updates and Maintenance

### Regular Tasks
- Monitor ML model performance
- Review and adjust risk settings
- Update dependencies
- Backup trading data

### Version Updates
- Check for new features and improvements
- Update both backend and frontend components
- Migrate database if needed

## 📞 Support

For technical support or questions:
- Check the troubleshooting section
- Review API documentation
- Contact development team

## ⚠️ Disclaimer

This trading bot is for educational and research purposes. Trading involves significant risk of loss. Never trade with money you cannot afford to lose. Past performance does not guarantee future results. Always test with demo accounts before using real money.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
