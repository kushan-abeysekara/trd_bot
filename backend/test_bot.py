#!/usr/bin/env python3
"""
Trading Bot Quick Setup and Test Script
This script helps set up and test the trading bot quickly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.trading_bot import trading_bot
from datetime import datetime
import time
import random

def setup_and_test_bot():
    """Set up and test the trading bot"""
    print("🚀 Trading Bot Quick Setup")
    print("=" * 40)
    
    # 1. Set up mock API token
    print("\n1. Setting up mock API token...")
    mock_api_token = "DEMO123456789"
    result = trading_bot.set_api_token(mock_api_token)
    print(f"   - API Token setup: {result}")
    
    # 2. Start the bot
    print("\n2. Starting the trading bot...")
    start_result = trading_bot.start()
    print(f"   - Start result: {start_result}")
    
    if start_result['success']:
        print("   ✅ Bot started successfully!")
        
        # 3. Generate some mock price data
        print("\n3. Generating mock market data...")
        base_price = 1.12345
        for i in range(100):
            # Generate realistic price movement
            change = random.uniform(-0.0005, 0.0005)
            base_price += change
            
            # Add some fake tick data to simulate real market
            tick_data = {
                'price': base_price,
                'timestamp': datetime.utcnow(),
                'symbol': 'R_10'
            }
            
            trading_bot._process_tick_data(tick_data)
            time.sleep(0.01)  # Small delay to simulate real-time data
        
        print(f"   ✅ Generated {len(trading_bot.price_history)} price points")
        
        # 4. Check bot status after data
        print("\n4. Checking bot status with data...")
        status = trading_bot.get_status()
        print(f"   - Current Price: ${status['current_price']:.5f}")
        print(f"   - Strategy Status: {status['strategy_status']}")
        print(f"   - Price History: {len(trading_bot.price_history)} points")
        
        # 5. Run for a few seconds to see if trades trigger
        print("\n5. Running bot for 10 seconds...")
        print("   - Watching for trading signals...")
        
        for i in range(100):  # Run for 10 seconds (100 * 0.1s)
            # Generate more price data
            change = random.uniform(-0.0002, 0.0002)
            base_price += change
            
            tick_data = {
                'price': base_price,
                'timestamp': datetime.utcnow(),
                'symbol': 'R_10'
            }
            
            trading_bot._process_tick_data(tick_data)
            
            # Check for new trades
            active_trades = trading_bot.get_active_trades()
            if active_trades:
                print(f"   🎯 Trade detected! {len(active_trades)} active trades")
                for trade in active_trades:
                    print(f"      - {trade['direction']} at ${trade['entry_price']:.5f}")
            
            time.sleep(0.1)
        
        # 6. Final status
        print("\n6. Final Status:")
        final_status = trading_bot.get_status()
        stats = trading_bot.get_statistics()
        
        print(f"   - Active Trades: {final_status['active_trades_count']}")
        print(f"   - Total Trades: {stats['total_trades']}")
        print(f"   - Won Trades: {stats['won_trades']}")
        print(f"   - Lost Trades: {stats['lost_trades']}")
        print(f"   - Net Profit: ${stats['net_profit']:.2f}")
        print(f"   - Current Strategy Status: {final_status['strategy_status']}")
        
        # 7. Stop the bot
        print("\n7. Stopping the bot...")
        stop_result = trading_bot.stop()
        print(f"   - Stop result: {stop_result}")
        
    else:
        print("   ❌ Failed to start bot")
    
    print("\n✅ Setup and test complete!")

if __name__ == "__main__":
    setup_and_test_bot()
