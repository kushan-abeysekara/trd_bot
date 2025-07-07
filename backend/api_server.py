from flask import Flask, jsonify, request
from trading_bot import TradingBot

app = Flask(__name__)

# Initialize the trading bot (replace with your actual API token)
trading_bot = TradingBot("YOUR_API_TOKEN_HERE")

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current status of the trading bot"""
    try:
        status = trading_bot.get_bot_status()
        return jsonify({"success": True, "status": status})
    except Exception as e:
        print(f"Error getting status: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Start automated trading"""
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
    try:
        trading_bot.stop_trading()
        return jsonify({"success": True, "message": "Trading stopped"})
    except Exception as e:
        print(f"Error stopping trading: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/reset-session-counter', methods=['POST'])
def reset_session_counter():
    """Reset the trading session counter"""
    try:
        if trading_bot and hasattr(trading_bot, 'reset_session_trade_counter'):
            trading_bot.reset_session_trade_counter()
            return jsonify({"success": True, "message": "Session trade counter reset successfully"})
        return jsonify({"success": False, "message": "Trading bot not initialized"}), 400
    except Exception as e:
        print(f"Error resetting session counter: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/disable-session-limit', methods=['POST'])
def disable_session_limit():
    """Disable the session trade limit"""
    try:
        if trading_bot and hasattr(trading_bot, 'set_risk_limits'):
            trading_bot.set_risk_limits(max_trades=0)  # Set to 0 to disable
            return jsonify({"success": True, "message": "Session trade limit disabled"})
        return jsonify({"success": False, "message": "Trading bot not initialized"}), 400
    except Exception as e:
        print(f"Error disabling session limit: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

# ...existing API endpoints...

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)