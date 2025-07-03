#!/usr/bin/env python3
"""
Check Dashboard Status
This script checks if both backend and frontend are running correctly.
"""

import requests
import json
import sys

def check_backend():
    """Check if backend is running"""
    try:
        # Test health endpoint
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is running")
            print(f"   - Status: {data.get('status')}")
            print(f"   - Database: {data.get('database')}")
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def check_trading_bot_api():
    """Check if trading bot API is working"""
    try:
        response = requests.get('http://localhost:5000/api/trading-bot/test', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Trading Bot API is working")
            bot_status = data.get('bot_status', {})
            print(f"   - Bot Running: {bot_status.get('is_running')}")
            print(f"   - Strategy: {bot_status.get('current_strategy')}")
            print(f"   - Status: {bot_status.get('strategy_status')}")
            return True
        else:
            print(f"❌ Trading Bot API error: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Trading Bot API not accessible: {e}")
        return False

def check_frontend():
    """Check if frontend is running"""
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is running")
            print("   - Dashboard: http://localhost:3000")
            return True
        else:
            print(f"❌ Frontend error: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend not accessible: {e}")
        return False

def main():
    print("🔍 Trading Bot System Status Check")
    print("=" * 40)
    
    backend_ok = check_backend()
    bot_api_ok = check_trading_bot_api()
    frontend_ok = check_frontend()
    
    print("\n📊 Summary:")
    if backend_ok and bot_api_ok and frontend_ok:
        print("✅ All systems operational!")
        print("\n🚀 Ready to trade:")
        print("   1. Open dashboard: http://localhost:3000")
        print("   2. Setup API token in dashboard")
        print("   3. Start the trading bot")
        print("   4. Monitor live trading activity")
    else:
        print("❌ Some systems need attention:")
        if not backend_ok:
            print("   - Start backend: cd backend && python app.py")
        if not frontend_ok:
            print("   - Start frontend: cd frontend && npm start")
        
    print(f"\n💡 Bot Working Status: {'NORMAL' if bot_api_ok else 'NEEDS RESTART'}")

if __name__ == "__main__":
    main()
