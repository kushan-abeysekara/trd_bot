#!/usr/bin/env python3
"""
Trading Bot Diagnostic Script
This script helps diagnose issues with the trading bot not starting or trading.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.trading_bot import trading_bot
import time
import json

def run_diagnostics():
    """Run comprehensive diagnostics on the trading bot"""
    print("🔍 Trading Bot Diagnostic Report")
    print("=" * 50)
    
    # 1. Check bot status
    print("\n1. Bot Status Check:")
    status = trading_bot.get_status()
    print(f"   - Is Running: {status['is_running']}")
    print(f"   - Is Connected: {status['is_connected']}")
    print(f"   - Current Strategy: {status['current_strategy']}")
    print(f"   - Strategy Status: {status['strategy_status']}")
    print(f"   - Active Trades: {status['active_trades_count']}")
    print(f"   - Account Balance: ${status['account_balance']}")
    print(f"   - Daily Profit: ${status['daily_profit']}")
    print(f"   - Daily Loss: ${status['daily_loss']}")
    print(f"   - Win Rate: {status['win_rate']}%")
    
    # 2. Check settings
    print("\n2. Bot Settings:")
    settings = status['settings']
    print(f"   - Auto Stake Enabled: {settings['auto_stake_enabled']}")
    print(f"   - Auto Stake: ${settings['auto_stake']}")
    print(f"   - Manual Stake: ${settings['manual_stake']}")
    print(f"   - Max Stake: ${settings['max_stake']}")
    print(f"   - Min Stake: ${settings['min_stake']}")
    print(f"   - Daily Stop Loss: ${settings['daily_stop_loss']}")
    print(f"   - Daily Target: ${settings['daily_target']}")
    print(f"   - Max Concurrent Trades: {settings['max_concurrent_trades']}")
    print(f"   - Strategy Mode: {settings['strategy_mode']}")
    
    # 3. Check technical indicators
    print("\n3. Technical Indicators:")
    print(f"   - Current Price: ${trading_bot.current_price}")
    print(f"   - RSI: {trading_bot.rsi}")
    print(f"   - MACD: {trading_bot.macd}")
    print(f"   - Bollinger Bands: {trading_bot.bollinger_bands}")
    print(f"   - Momentum: {trading_bot.momentum}")
    print(f"   - Volatility: {trading_bot.volatility}")
    
    # 4. Check price history
    print("\n4. Market Data:")
    print(f"   - Price History Length: {len(trading_bot.price_history)}")
    print(f"   - Tick Data Length: {len(trading_bot.tick_data)}")
    if len(trading_bot.price_history) > 0:
        print(f"   - Latest Price: ${trading_bot.price_history[-1]}")
    
    # 5. Check trading conditions
    print("\n5. Trading Conditions Analysis:")
    can_trade = trading_bot._can_place_trade()
    print(f"   - Can Place Trade: {can_trade}")
    
    if not can_trade:
        print("   - Reasons why trading is blocked:")
        if abs(trading_bot.daily_loss) >= trading_bot.settings.daily_stop_loss:
            print(f"     ❌ Daily stop loss hit: ${abs(trading_bot.daily_loss)} >= ${trading_bot.settings.daily_stop_loss}")
        if trading_bot.daily_profit >= trading_bot.settings.daily_target:
            print(f"     ❌ Daily target reached: ${trading_bot.daily_profit} >= ${trading_bot.settings.daily_target}")
        if len(trading_bot.active_trades) >= trading_bot.settings.max_concurrent_trades:
            print(f"     ❌ Max concurrent trades: {len(trading_bot.active_trades)} >= {trading_bot.settings.max_concurrent_trades}")
        if trading_bot.last_trade_time:
            time_since_last = (trading_bot.datetime.utcnow() - trading_bot.last_trade_time).total_seconds()
            if time_since_last < trading_bot.trade_cooldown:
                print(f"     ❌ Trade cooldown: {time_since_last}s < {trading_bot.trade_cooldown}s")
    
    # 6. Check strategy conditions
    print("\n6. Strategy Conditions (Adaptive Mean Reversion):")
    if len(trading_bot.price_history) >= 50:
        rsi_ok = 48 <= trading_bot.rsi <= 52
        volatility_percent = trading_bot.volatility * 100
        volatility_ok = 1.0 <= volatility_percent <= 1.5
        momentum_percent = abs(trading_bot.momentum * 100)
        momentum_ok = momentum_percent < 0.2
        macd_ok = -0.1 <= trading_bot.macd['macd'] <= 0.1
        
        print(f"   - RSI in range (48-52): {rsi_ok} (Current: {trading_bot.rsi:.2f})")
        print(f"   - Volatility in range (1-1.5%): {volatility_ok} (Current: {volatility_percent:.2f}%)")
        print(f"   - Momentum < 0.2%: {momentum_ok} (Current: {momentum_percent:.2f}%)")
        print(f"   - MACD flat (-0.1 to +0.1): {macd_ok} (Current: {trading_bot.macd['macd']:.4f})")
        
        if rsi_ok and volatility_ok and momentum_ok and macd_ok:
            print("   ✅ All strategy conditions met - Waiting for Bollinger Band touch")
        else:
            print("   ❌ Strategy conditions not met")
    else:
        print("   ❌ Insufficient price history (need >= 50 points)")
    
    # 7. Test bot start/stop
    print("\n7. Bot Control Test:")
    if not trading_bot.is_running:
        print("   - Attempting to start bot...")
        result = trading_bot.start()
        print(f"   - Start result: {result}")
        time.sleep(2)
        
        print("   - Checking if bot started...")
        status = trading_bot.get_status()
        print(f"   - Bot running: {status['is_running']}")
        
        if status['is_running']:
            print("   ✅ Bot started successfully")
            time.sleep(3)
            print("   - Stopping bot...")
            stop_result = trading_bot.stop()
            print(f"   - Stop result: {stop_result}")
        else:
            print("   ❌ Bot failed to start")
    else:
        print("   - Bot is already running")
    
    # 8. API Token check
    print("\n8. API Configuration:")
    if trading_bot.api_token:
        print(f"   - API Token: {trading_bot.api_token[:10]}...***")
    else:
        print("   ❌ No API token configured (using mock data)")
    
    # 9. Final recommendations
    print("\n9. Recommendations:")
    if not trading_bot.is_running:
        print("   📋 To start trading:")
        print("   1. Ensure API token is configured")
        print("   2. Start the bot via the dashboard or API")
        print("   3. Wait for sufficient price data (50+ points)")
        print("   4. Ensure strategy conditions are met")
    else:
        print("   📋 Bot is running but not trading:")
        print("   1. Check strategy conditions above")
        print("   2. Ensure daily limits not reached")
        print("   3. Wait for Bollinger Band touches")
        print("   4. Monitor real-time data feed")
    
    print("\n✅ Diagnostic complete!")

if __name__ == "__main__":
    run_diagnostics()
