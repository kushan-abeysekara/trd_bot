from flask import Flask, jsonify, request
from trading_bot import TradingBot

app = Flask(__name__)

# Initialize the trading bot as None - will be set when API token is provided
trading_bot = None

@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect to Deriv API with provided token"""
    global trading_bot
    
    data = request.json
    api_token = data.get('api_token')
    
    if not api_token:
        return jsonify({"success": False, "message": "API token is required"}), 400
    
    try:
        # Initialize trading bot with the provided token
        trading_bot = TradingBot(api_token)
        trading_bot.connect()
        return jsonify({"success": True, "message": "Connected to Deriv API"})
    except Exception as e:
        print(f"Error connecting: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current status of the trading bot"""
    if not trading_bot:
        return jsonify({"success": False, "message": "Trading bot not connected"}), 400
    
    try:
        status = trading_bot.get_bot_status()
        return jsonify({"success": True, "status": status})
    except Exception as e:
        print(f"Error getting status: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/balance', methods=['GET'])
def get_balance():
    """Get current balance"""
    if not trading_bot:
        return jsonify({"success": False, "message": "Trading bot not connected"}), 400
    
    try:
        balance = trading_bot.get_balance()
        return jsonify({"success": True, "balance": balance})
    except Exception as e:
        print(f"Error getting balance: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Start automated trading"""
    if not trading_bot:
        return jsonify({"success": False, "message": "Trading bot not connected"}), 400
    
    data = request.json
    amount = data.get('amount', 1.0)
    duration = data.get('duration', 1)
    
    try:
        trading_bot.start_trading(amount, duration)
        return jsonify({"success": True, "message": "Trading started"})
    except Exception as e:
        print(f"Error starting trading: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/stop-trading', methods=['POST'])
def stop_trading():
    """Stop automated trading"""
    if not trading_bot:
        return jsonify({"success": False, "message": "Trading bot not connected"}), 400
    
    try:
        trading_bot.stop_trading()
        return jsonify({"success": True, "message": "Trading stopped"})
    except Exception as e:
        print(f"Error stopping trading: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/reset-session-counter', methods=['POST'])
def reset_session_counter():
    """Reset the trading session counter"""
    if not trading_bot:
        return jsonify({"success": False, "message": "Trading bot not connected"}), 400
    
    try:
        trading_bot.reset_session_trade_counter()
        return jsonify({"success": True, "message": "Session trade counter reset successfully"})
    except Exception as e:
        print(f"Error resetting session counter: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/disable-session-limit', methods=['POST'])
def disable_session_limit():
    """Disable the session trade limit"""
    if not trading_bot:
        return jsonify({"success": False, "message": "Trading bot not connected"}), 400
    
    try:
        trading_bot.set_risk_limits(max_trades=0)  # Set to 0 to disable
        return jsonify({"success": True, "message": "Session trade limit disabled"})
    except Exception as e:
        print(f"Error disabling session limit: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

# ...existing code...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)