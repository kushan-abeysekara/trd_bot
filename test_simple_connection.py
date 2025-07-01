#!/usr/bin/env python3
"""
Simple Deriv API Connection Test
"""
import asyncio
import websockets
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_simple_connection():
    """Test basic connection without auth first"""
    
    endpoints = [
        "wss://ws.binaryws.com/websockets/v3",
        "wss://frontend.binaryws.com/websockets/v3", 
        "wss://ws.derivws.com/websockets/v3"
    ]
    
    for ws_url in endpoints:
        try:
            logger.info(f"Testing connection to: {ws_url}")
            
            # Connect to WebSocket
            websocket = await websockets.connect(ws_url)
            logger.info("✅ Connected successfully!")
            
            # Test ping
            ping_request = {
                "ping": 1,
                "req_id": 1
            }
            
            await websocket.send(json.dumps(ping_request))
            logger.info("📤 Ping sent")
            
            # Wait for response
            response = await websockets.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            
            logger.info(f"📨 Response: {data}")
            
            if 'pong' in data:
                logger.info("✅ Ping successful!")
                
                # Now test with your token
                token = "B8ZH857zyOqvHah"
                auth_request = {
                    "authorize": token,
                    "req_id": 2
                }
                
                await websocket.send(json.dumps(auth_request))
                auth_response = await websockets.wait_for(websocket.recv(), timeout=10)
                auth_data = json.loads(auth_response)
                
                logger.info(f"🔐 Auth response: {auth_data}")
                
                if 'authorize' in auth_data:
                    logger.info("🎉 AUTHENTICATION SUCCESSFUL!")
                    logger.info(f"Account: {auth_data['authorize'].get('loginid', 'N/A')}")
                    logger.info(f"Currency: {auth_data['authorize'].get('currency', 'N/A')}")
                    logger.info(f"Balance: {auth_data['authorize'].get('balance', 'N/A')}")
                    
                    await websocket.close()
                    return ws_url, True
                    
                elif 'error' in auth_data:
                    logger.error(f"❌ Auth failed: {auth_data['error']}")
                    
            await websocket.close()
            
        except Exception as e:
            logger.error(f"❌ Failed for {ws_url}: {e}")
            continue
    
    return None, False

async def main():
    print("🔧 Testing Deriv API Endpoints...")
    print("=" * 60)
    
    working_url, success = await test_simple_connection()
    
    if success:
        print(f"\n🎉 SUCCESS! Working endpoint: {working_url}")
        print("✅ Your token is valid and working!")
        print("✅ Ready to start the trading bot!")
    else:
        print("\n❌ No working endpoints found")
        print("🔧 Please check:")
        print("1. Internet connection")
        print("2. Token validity at https://app.deriv.com/account/api-token")
        print("3. Token permissions (Read, Trade)")

if __name__ == "__main__":
    asyncio.run(main())
