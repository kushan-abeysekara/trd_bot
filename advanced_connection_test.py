#!/usr/bin/env python3
"""
Advanced Deriv API Connection Test with Multiple URLs
"""
import asyncio
import websockets
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_connection_urls():
    """Test multiple Deriv WebSocket URLs"""
    
    token = "oYg6w0KcXrnrxHF"
    
    # Different possible URLs for Deriv API
    test_urls = [
        "wss://ws.binaryws.com/websockets/v3",
        "wss://ws.derivws.com/websockets/v3", 
        "wss://frontend.binaryws.com/websockets/v3",
        "wss://green.binaryws.com/websockets/v3",
        "wss://blue.binaryws.com/websockets/v3",
        "wss://red.binaryws.com/websockets/v3",
        "wss://ws.binaryws.com/websockets/v3?app_id=1089",
        "wss://ws.binaryws.com/websockets/v3?app_id=1089&l=EN&brand=deriv"
    ]
    
    for url in test_urls:
        print(f"\n🔄 Testing: {url}")
        print("-" * 60)
        
        try:
            # Test basic connection
            websocket = await websockets.connect(url)
            print(f"✅ Connection successful to: {url}")
            
            # Test ping first
            ping_request = {"ping": 1, "req_id": 1}
            await websocket.send(json.dumps(ping_request))
            
            # Wait for pong
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            
            if 'pong' in data:
                print("✅ Ping successful")
                
                # Now try authorization
                auth_request = {"authorize": token, "req_id": 2}
                await websocket.send(json.dumps(auth_request))
                
                auth_response = await asyncio.wait_for(websocket.recv(), timeout=10)
                auth_data = json.loads(auth_response)
                
                print(f"📨 Auth response: {auth_data}")
                
                if 'error' in auth_data:
                    print(f"❌ Auth failed: {auth_data['error']}")
                elif 'authorize' in auth_data:
                    print("🎉 AUTHORIZATION SUCCESSFUL!")
                    print(f"Account: {auth_data['authorize'].get('loginid', 'N/A')}")
                    print(f"Currency: {auth_data['authorize'].get('currency', 'N/A')}")
                    print(f"Balance: {auth_data['authorize'].get('balance', 'N/A')}")
                    
                    # Test getting account status
                    balance_request = {"balance": 1, "subscribe": 1, "req_id": 3}
                    await websocket.send(json.dumps(balance_request))
                    
                    balance_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    balance_data = json.loads(balance_response)
                    print(f"💰 Balance info: {balance_data}")
                    
                    await websocket.close()
                    return url  # Return successful URL
                    
            await websocket.close()
            
        except asyncio.TimeoutError:
            print(f"⏱️  Timeout connecting to: {url}")
        except Exception as e:
            print(f"❌ Failed to connect to: {url}")
            print(f"   Error: {e}")
    
    return None

async def test_market_data(working_url):
    """Test market data fetching"""
    if not working_url:
        print("❌ No working URL found for market data test")
        return
        
    print(f"\n📊 Testing market data with: {working_url}")
    print("-" * 60)
    
    token = "oYg6w0KcXrnrxHF"
    
    try:
        websocket = await websockets.connect(working_url)
        
        # Authorize first
        auth_request = {"authorize": token, "req_id": 1}
        await websocket.send(json.dumps(auth_request))
        await websocket.recv()  # Skip auth response
        
        # Get available symbols
        symbols_request = {"active_symbols": "brief", "product_type": "basic", "req_id": 2}
        await websocket.send(json.dumps(symbols_request))
        
        symbols_response = await websocket.recv()
        symbols_data = json.loads(symbols_response)
        
        if 'active_symbols' in symbols_data:
            volatility_indices = [s for s in symbols_data['active_symbols'] 
                                if s['symbol'].startswith('R_') and s['market'] == 'synthetic_index']
            
            print(f"✅ Found {len(volatility_indices)} volatility indices:")
            for symbol in volatility_indices[:5]:  # Show first 5
                print(f"   - {symbol['symbol']}: {symbol['display_name']}")
        
        # Test tick subscription for R_10
        tick_request = {
            "ticks": "R_10",
            "subscribe": 1,
            "req_id": 3
        }
        await websocket.send(json.dumps(tick_request))
        
        print("\n📈 Listening for R_10 ticks (will show 3 ticks)...")
        
        tick_count = 0
        while tick_count < 3:
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(response)
            
            if 'tick' in data:
                tick = data['tick']
                print(f"   Tick {tick_count + 1}: {tick['symbol']} = {tick['quote']} at {tick['epoch']}")
                tick_count += 1
        
        await websocket.close()
        print("✅ Market data test successful!")
        
    except Exception as e:
        print(f"❌ Market data test failed: {e}")

async def main():
    print("🚀 ADVANCED DERIV API CONNECTION TEST")
    print("=" * 60)
    
    # Test connection URLs
    working_url = await test_connection_urls()
    
    if working_url:
        print(f"\n🎉 SUCCESS! Working URL found: {working_url}")
        
        # Test market data
        await test_market_data(working_url)
        
        print(f"\n✅ UPDATE YOUR .env FILE:")
        print(f"DERIV_WS_URL={working_url}")
        
    else:
        print("\n❌ NO WORKING URLs FOUND!")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Verify your token is active at https://app.deriv.com/account/api-token")
        print("2. Check if your demo account has sufficient balance")
        print("3. Ensure token has 'Read' and 'Trade' permissions")
        print("4. Try generating a new token")

if __name__ == "__main__":
    asyncio.run(main())
