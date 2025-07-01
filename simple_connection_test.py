#!/usr/bin/env python3
"""
Simple WebSocket Test - No Auth Required
"""
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_connection():
    """Test basic WebSocket connection and get market data without auth"""
    
    url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    
    print(f"🔄 Testing basic connection to: {url}")
    print("-" * 60)
    
    try:
        websocket = await websockets.connect(url)
        print("✅ Connected successfully!")
        
        # Test ping
        ping_request = {"ping": 1, "req_id": 1}
        await websocket.send(json.dumps(ping_request))
        
        response = await asyncio.wait_for(websocket.recv(), timeout=5)
        data = json.loads(response)
        print(f"📨 Ping response: {data}")
        
        # Get server time (no auth needed)
        time_request = {"time": 1, "req_id": 2}
        await websocket.send(json.dumps(time_request))
        
        time_response = await websocket.recv()
        time_data = json.loads(time_response)
        print(f"🕐 Server time: {time_data}")
        
        # Get active symbols (no auth needed)
        symbols_request = {"active_symbols": "brief", "product_type": "basic", "req_id": 3}
        await websocket.send(json.dumps(symbols_request))
        
        symbols_response = await websocket.recv()
        symbols_data = json.loads(symbols_response)
        
        if 'active_symbols' in symbols_data:
            volatility_indices = [s for s in symbols_data['active_symbols'] 
                                if s['symbol'].startswith('R_')]
            
            print(f"✅ Found {len(volatility_indices)} volatility indices:")
            for symbol in volatility_indices[:10]:
                print(f"   - {symbol['symbol']}: {symbol['display_name']}")
        
        # Subscribe to R_10 ticks (no auth needed for demo)
        tick_request = {"ticks": "R_10", "subscribe": 1, "req_id": 4}
        await websocket.send(json.dumps(tick_request))
        
        print(f"\n📈 Getting R_10 ticks...")
        
        # Get a few ticks
        for i in range(3):
            response = await asyncio.wait_for(websocket.recv(), timeout=15)
            data = json.loads(response)
            
            if 'tick' in data:
                tick = data['tick']
                print(f"   Tick {i+1}: {tick['symbol']} = {tick['quote']} at {tick['epoch']}")
            elif 'subscription' in data:
                print(f"   Subscription confirmed: {data['subscription']}")
        
        await websocket.close()
        print("✅ Basic connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

async def test_with_auth():
    """Test with authorization using different token formats"""
    
    url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    token = "oYg6w0KcXrnrxHF"
    
    print(f"\n🔐 Testing authorization...")
    print("-" * 60)
    
    try:
        websocket = await websockets.connect(url)
        
        # Try different auth formats
        auth_formats = [
            {"authorize": token},
            {"authorize": token, "req_id": 1},
            {"api_token": token, "req_id": 2},
            {"oauth_token": token, "req_id": 3}
        ]
        
        for i, auth_request in enumerate(auth_formats):
            print(f"   Trying format {i+1}: {list(auth_request.keys())}")
            
            await websocket.send(json.dumps(auth_request))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                print(f"   Response: {data}")
                
                if 'authorize' in data and 'loginid' in data['authorize']:
                    print("🎉 AUTHORIZATION SUCCESSFUL!")
                    print(f"   Account: {data['authorize']['loginid']}")
                    print(f"   Currency: {data['authorize']['currency']}")
                    print(f"   Balance: {data['authorize']['balance']}")
                    await websocket.close()
                    return True
                elif 'error' in data:
                    print(f"   ❌ Error: {data['error']['message']}")
                    
            except asyncio.TimeoutError:
                print(f"   ⏱️  No response for format {i+1}")
        
        await websocket.close()
        return False
        
    except Exception as e:
        print(f"❌ Auth test failed: {e}")
        return False

async def main():
    print("🧪 SIMPLE DERIV API TEST")
    print("=" * 50)
    
    # Test basic connection first
    basic_success = await test_basic_connection()
    
    if basic_success:
        print("\n✅ Basic connection works! WebSocket is functional.")
        
        # Test authorization
        auth_success = await test_with_auth()
        
        if auth_success:
            print("\n🎉 FULL SUCCESS! Ready to start trading!")
            print("\n🚀 Next steps:")
            print("1. Update .env file with working URL")
            print("2. Start the trading bot")
            print("3. Monitor performance on dashboard")
        else:
            print("\n⚠️  Authorization failed, but we can still get market data")
            print("   The bot can run in read-only mode for testing")
    else:
        print("\n❌ Basic connection failed")
        print("   Check your internet connection and firewall")

if __name__ == "__main__":
    asyncio.run(main())
