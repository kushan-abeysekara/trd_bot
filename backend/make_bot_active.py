#!/usr/bin/env python3
"""
Quick Fix: Make Trading Bot More Active
This script temporarily adjusts the bot settings to make it trade more frequently for testing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.trading_bot import trading_bot

def make_bot_more_active():
    """Adjust bot settings to make it trade more frequently"""
    print("🔧 Adjusting Trading Bot for More Activity")
    print("=" * 50)
    
    # Get current status
    status = trading_bot.get_status()
    print(f"Current Strategy Status: {status['strategy_status']}")
    
    # Update settings to be more aggressive
    new_settings = {
        'auto_stake': 0.35,  # Lower stake for testing
        'min_stake': 0.35,
        'max_stake': 5.0,
        'daily_stop_loss': 20.0,  # Lower stop loss for testing
        'daily_target': 10.0,     # Lower target for testing
        'max_concurrent_trades': 5,  # More concurrent trades
        'strategy_mode': 'AGGRESSIVE'  # More aggressive mode
    }
    
    print("\n📝 Updating bot settings for more activity...")
    result = trading_bot.update_settings(new_settings)
    print(f"Settings update: {result}")
    
    # Temporarily modify strategy conditions for testing
    print("\n⚙️ Temporarily relaxing strategy conditions...")
    
    # Override the _check_trading_signals method to be more lenient
    original_check_signals = trading_bot._check_trading_signals
    
    def relaxed_check_signals():
        """More lenient trading signal detection"""
        if len(trading_bot.price_history) < 20:
            return None
        
        # Much more relaxed conditions
        rsi_ok = 30 <= trading_bot.rsi <= 70  # Relaxed from 48-52
        volatility_percent = trading_bot.volatility * 100
        volatility_ok = 0.5 <= volatility_percent <= 3.0  # Relaxed from 1-1.5%
        momentum_percent = abs(trading_bot.momentum * 100)
        momentum_ok = momentum_percent < 1.0  # Relaxed from 0.2%
        
        if rsi_ok and volatility_ok and momentum_ok:
            # Simple signal based on recent price movement
            if len(trading_bot.tick_data) >= 3:
                recent_prices = [t['price'] for t in list(trading_bot.tick_data)[-3:]]
                if recent_prices[-1] > recent_prices[-2]:  # Price going up
                    return {
                        'direction': trading_bot.TradeDirection.RISE,
                        'confidence': 60,
                        'reason': 'Relaxed conditions + Rising price',
                        'entry_price': trading_bot.current_price
                    }
                elif recent_prices[-1] < recent_prices[-2]:  # Price going down
                    return {
                        'direction': trading_bot.TradeDirection.FALL,
                        'confidence': 60,
                        'reason': 'Relaxed conditions + Falling price',
                        'entry_price': trading_bot.current_price
                    }
        
        return None
    
    # Temporarily replace the method
    trading_bot._check_trading_signals = relaxed_check_signals
    
    print("✅ Bot configured for more active trading!")
    print("\n📋 New conditions:")
    print("   - RSI: 30-70 (was 48-52)")
    print("   - Volatility: 0.5-3% (was 1-1.5%)")
    print("   - Momentum: <1% (was <0.2%)")
    print("   - Simple trend following signals")
    
    print(f"\n🚀 Ready to trade! Start the bot with:")
    print(f"   trading_bot.start()")
    
    return original_check_signals

def restore_original_settings(original_check_signals):
    """Restore original conservative settings"""
    print("\n🔄 Restoring original conservative settings...")
    
    # Restore original method
    trading_bot._check_trading_signals = original_check_signals
    
    # Restore conservative settings
    original_settings = {
        'auto_stake': 1.0,
        'min_stake': 0.35,
        'max_stake': 10.0,
        'daily_stop_loss': 50.0,
        'daily_target': 20.0,
        'max_concurrent_trades': 3,
        'strategy_mode': 'ADAPTIVE'
    }
    
    trading_bot.update_settings(original_settings)
    print("✅ Original settings restored")

if __name__ == "__main__":
    import time
    
    # Make bot more active
    original_check_signals = make_bot_more_active()
    
    print("\n" + "="*50)
    print("🎯 Bot is now configured for active trading!")
    print("Run this to start trading:")
    print("   python -c \"from services.trading_bot import trading_bot; trading_bot.start(); import time; time.sleep(60); trading_bot.stop()\"")
    print("="*50)
