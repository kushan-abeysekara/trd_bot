#!/usr/bin/env python3
"""
Trading Bot Error Fixer
This script diagnoses and fixes common issues with the trading bot.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.trading_bot import trading_bot
import time
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_bot_errors():
    """Fix common errors in the trading bot"""
    print("🔧 Trading Bot Error Fixer")
    print("=" * 50)
    
    # 1. Check if the bot exists and has required attributes
    print("\n1. Checking bot initialization...")
    
    # Fix strategy_stats attribute
    if not hasattr(trading_bot, 'strategy_stats') or trading_bot.strategy_stats is None:
        print("   ❌ Missing 'strategy_stats' attribute - Fixing...")
        trading_bot.strategy_stats = {}
        
        # Initialize with default values for all available strategies
        strategies = trading_bot.get_available_strategies()
        for strategy in strategies:
            strategy_id = strategy['id']
            trading_bot.strategy_stats[strategy_id] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'profit': 0.0,
                'status': 'Inactive',
                'last_signal_time': None
            }
        print("   ✅ 'strategy_stats' attribute initialized")
    else:
        print("   ✅ 'strategy_stats' attribute exists")
    
    # Fix active_strategies attribute
    if not hasattr(trading_bot, 'active_strategies') or trading_bot.active_strategies is None:
        print("   ❌ Missing 'active_strategies' attribute - Fixing...")
        trading_bot.active_strategies = {}
        
        # Initialize with default values
        strategies = trading_bot.get_available_strategies()
        for i, strategy in enumerate(strategies):
            strategy_id = strategy['id']
            trading_bot.active_strategies[strategy_id] = (i == 0)  # Enable only the first one
        print("   ✅ 'active_strategies' attribute initialized")
    else:
        print("   ✅ 'active_strategies' attribute exists")
    
    # Fix strategy_trades attribute
    if not hasattr(trading_bot, 'strategy_trades') or trading_bot.strategy_trades is None:
        print("   ❌ Missing 'strategy_trades' attribute - Fixing...")
        trading_bot.strategy_trades = {}
        
        # Initialize with default values
        strategies = trading_bot.get_available_strategies()
        for strategy in strategies:
            strategy_id = strategy['id']
            trading_bot.strategy_trades[strategy_id] = []
        print("   ✅ 'strategy_trades' attribute initialized")
    else:
        print("   ✅ 'strategy_trades' attribute exists")
    
    # 2. Fix trade history issues
    print("\n2. Checking trade history...")
    
    if not hasattr(trading_bot, 'trade_history') or trading_bot.trade_history is None:
        print("   ❌ Missing 'trade_history' attribute - Fixing...")
        from collections import deque
        trading_bot.trade_history = deque(maxlen=1000)
        print("   ✅ 'trade_history' attribute initialized")
    else:
        print(f"   ✅ 'trade_history' attribute exists with {len(trading_bot.trade_history)} entries")
    
    # 3. Check technical indicators
    print("\n3. Checking technical indicators...")
    
    for attr in ['rsi', 'macd', 'bollinger_bands', 'momentum', 'volatility']:
        if not hasattr(trading_bot, attr):
            print(f"   ❌ Missing '{attr}' attribute - Fixing...")
            if attr == 'rsi':
                trading_bot.rsi = 50.0
            elif attr == 'macd':
                trading_bot.macd = {'macd': 0, 'signal': 0, 'histogram': 0}
            elif attr == 'bollinger_bands':
                trading_bot.bollinger_bands = {'upper': 0, 'middle': 0, 'lower': 0}
            elif attr == 'momentum':
                trading_bot.momentum = 0.0
            elif attr == 'volatility':
                trading_bot.volatility = 0.0
            print(f"   ✅ '{attr}' attribute initialized")
        else:
            print(f"   ✅ '{attr}' attribute exists")
    
    # 4. Initialize macd_history if missing
    if not hasattr(trading_bot, 'macd_history'):
        print("   ❌ Missing 'macd_history' attribute - Fixing...")
        from collections import deque
        trading_bot.macd_history = deque(maxlen=100)
        print("   ✅ 'macd_history' attribute initialized")
    
    # 5. Check API token and WebSocket connection
    print("\n4. Checking API connection...")
    
    if not trading_bot.api_token:
        print("   ❌ No API token configured - Using mock data")
    else:
        print(f"   ✅ API token configured (token length: {len(trading_bot.api_token)})")
    
    # 6. Restart the bot if it's in an inconsistent state
    print("\n5. Resetting bot state...")
    
    if trading_bot.is_running:
        print("   ⚠️ Bot is currently running - Stopping first...")
        trading_bot.stop()
        time.sleep(1)
    
    # Reset any inconsistent state
    trading_bot.is_running = False
    trading_bot.is_connected = False
    
    if trading_bot.ws:
        try:
            trading_bot.ws.close()
        except:
            pass
        trading_bot.ws = None
    
    print("   ✅ Bot state reset successfully")
    
    # 7. Final checks and summary
    print("\n6. Final checks...")
    
    # Check if _initialize_strategies method exists and call it if needed
    if hasattr(trading_bot, '_initialize_strategies') and callable(trading_bot._initialize_strategies):
        print("   ✅ Reinitializing strategies...")
        trading_bot._initialize_strategies()
    
    # Summary
    print("\n✅ Bot error fixes complete!")
    print("\n📋 Summary:")
    print("   - Fixed missing strategy_stats attribute")
    print("   - Fixed missing active_strategies attribute")
    print("   - Fixed missing strategy_trades attribute")
    print("   - Ensured trade_history is initialized")
    print("   - Initialized technical indicators")
    print("   - Reset bot state for clean startup")
    
    print("\n📌 Next steps:")
    print("   1. Start the bot using the dashboard")
    print("   2. Check the dashboard for live monitoring")
    print("   3. Monitor trading conditions for activity")
    
    print("\n⚠️ For real trading (not demo):")
    print("   1. Ensure valid API token is configured")
    print("   2. Start with small stakes")
    print("   3. Monitor first few trades carefully")

if __name__ == "__main__":
    fix_bot_errors()
    
    # Ask if user wants to start the bot
    print("\nDo you want to start the bot now? (y/n)")
    choice = input().lower()
    
    if choice == 'y':
        print("\nStarting trading bot...")
        result = trading_bot.start()
        print(f"Start result: {result}")
        
        if result['success']:
            print("\nBot started successfully!")
            print("Press Ctrl+C to stop the bot...")
            try:
                while True:
                    status = trading_bot.get_status()
                    print(f"Bot status: Running={status['is_running']}, "
                          f"Connected={status['is_connected']}, "
                          f"Active Trades={status['active_trades_count']}")
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\nStopping bot...")
                trading_bot.stop()
                print("Bot stopped.")
    else:
        print("\nBot not started. You can start it later using the dashboard.")
