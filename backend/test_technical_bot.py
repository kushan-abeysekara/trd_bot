#!/usr/bin/env python3
"""Test script to debug technical trading bot issues"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from routes.technical_trading_bot import TechnicalTradingBot
from utils.deriv_service import DerivService
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_technical_bot():
    """Test the technical trading bot initialization and startup"""
    
    print("Testing Technical Trading Bot...")
    
    # Test 1: Create TechnicalTradingBot instance
    try:
        bot = TechnicalTradingBot(user_id=1, api_token="dummy_token", account_type="demo")
        print("✓ TechnicalTradingBot instance created successfully")
    except Exception as e:
        print(f"✗ Failed to create TechnicalTradingBot: {str(e)}")
        return
    
    # Test 2: Check if database is initialized
    try:
        import sqlite3
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='technical_bot_sessions'")
        if cursor.fetchone():
            print("✓ Technical bot database tables exist")
        else:
            print("✗ Technical bot database tables missing")
        conn.close()
    except Exception as e:
        print(f"✗ Database check failed: {str(e)}")
    
    # Test 3: Check services initialization
    try:
        if hasattr(bot, 'technical_analyzer') and bot.technical_analyzer:
            print("✓ TechnicalAnalyzer initialized")
        else:
            print("✗ TechnicalAnalyzer not initialized")
            
        if hasattr(bot, 'strategy_engine') and bot.strategy_engine:
            print("✓ StrategyEngine initialized")
        else:
            print("✗ StrategyEngine not initialized")
            
        if hasattr(bot, 'deriv_service') and bot.deriv_service:
            print("✓ DerivService initialized")
        else:
            print("✗ DerivService not initialized")
    except Exception as e:
        print(f"✗ Service initialization check failed: {str(e)}")
    
    # Test 4: Try to start bot with dummy token (should fail gracefully)
    try:
        result = await bot.start_bot()
        if result['success']:
            print("✗ Bot started with dummy token (should have failed)")
        else:
            print(f"✓ Bot correctly failed to start with dummy token: {result['message']}")
    except Exception as e:
        print(f"✗ Exception during bot start test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_technical_bot())
