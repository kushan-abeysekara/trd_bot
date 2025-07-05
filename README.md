# Deriv Volatility Index Trading Bot

A full-stack automated trading bot for Deriv's Volatility Index with Python backend and React.js frontend.

## Features

- 🤖 Automated trading for Volatility 100 Index
- 📈 Rise/Fall contract trading (5 or 10 ticks)
- 💰 Real-time balance display
- 📊 Live trading statistics (win rate, profit/loss, trade count)
- 📱 Modern responsive web interface
- 🔒 Secure API token-based authentication
- 📝 Complete trade history tracking

## Project Structure

```
bot2/
├── backend/          # Python Flask backend
│   ├── app.py        # Main Flask application
│   ├── deriv_api.py  # Deriv WebSocket API wrapper
│   ├── trading_bot.py # Trading bot logic
│   └── requirements.txt # Python dependencies
├── frontend/         # React.js frontend
│   ├── public/       # Public assets
│   ├── src/          # React source code
│   └── package.json  # Node.js dependencies
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn
- Deriv API token (get from https://app.deriv.com/account/api-token)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```bash
   python app.py
   ```

The backend will start on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the React development server:
   ```bash
   npm start
   ```

The frontend will start on http://localhost:3000

## Usage

1. **Get Your API Token**: 
   - Go to https://app.deriv.com/account/api-token
   - Create a new token with trading permissions
   - Copy the token

2. **Start the Application**:
   - Ensure both backend and frontend servers are running
   - Open http://localhost:3000 in your browser

3. **Connect to Deriv**:
   - Enter your API token in the text box
   - Click "Connect to Deriv API"
   - Wait for successful connection

4. **Configure Trading**:
   - Set your trade amount (minimum $1)
   - Choose duration (5 or 10 ticks)
   - Click "Start Trading" to begin automated trading

5. **Monitor Trading**:
   - View real-time balance updates
   - Track trading statistics (win rate, profit/loss)
   - Monitor individual trades in the trade history

## Trading Logic

- The bot automatically places Rise/Fall trades on Volatility 100 Index
- Trades are randomly distributed between Rise (CALL) and Fall (PUT)
- Each trade has a configurable duration (5 or 10 ticks)
- The bot waits 30 seconds between trades
- Trade results are simulated with a 60% win rate for demonstration

## API Endpoints

### Backend REST API

- `POST /api/connect` - Connect to Deriv API
- `POST /api/start-trading` - Start automated trading
- `POST /api/stop-trading` - Stop automated trading
- `GET /api/balance` - Get current balance
- `GET /api/stats` - Get trading statistics
- `GET /api/trade-history` - Get trade history
- `POST /api/disconnect` - Disconnect from API

### WebSocket Events

- `connection_status` - Connection status updates
- `balance_update` - Real-time balance updates
- `trade_update` - Trade status updates
- `stats_update` - Statistics updates

## Security Notes

- Never share your API token
- Use tokens with minimal required permissions
- This is a demo application - use caution with real money
- The trading logic is simplified for demonstration purposes

## Customization

### Trading Parameters

Edit `trading_bot.py` to modify:
- Trade intervals (currently 30 seconds)
- Win rate simulation (currently 60%)
- Payout rates
- Trading symbols

### UI Customization

Edit React components in `frontend/src/` to modify:
- Styling and colors
- Layout and components
- Additional features

## Troubleshooting

### Common Issues

1. **Connection Failed**: 
   - Check your API token
   - Ensure you have trading permissions
   - Verify internet connection

2. **Backend Not Starting**:
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+)

3. **Frontend Not Loading**:
   - Install dependencies: `npm install`
   - Check Node.js version (16+)

### Support

For issues with:
- Deriv API: Check Deriv's official documentation
- Technical problems: Review the console logs in both backend and frontend

## Disclaimer

This is an educational project for demonstration purposes. Trading involves risk, and you should never trade with money you cannot afford to lose. The developers are not responsible for any financial losses incurred while using this bot.

## License

This project is for educational purposes only. Please comply with Deriv's terms of service when using their API.
