#!/usr/bin/env python3
"""
Deriv API Connection with Token in URL
"""
import asyncio
import websockets
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_with_token_in_url():
    """Test connection with token in URL"""
    
    token = "oYg6w0KcXrnrxHF"  # Updated to match .env file
    
    # Different URL formats to try
    urls_to_try = [
        f"wss://ws.binaryws.com/websockets/v3?app_id=1089&l=EN&brand=deriv",
        f"wss://ws.binaryws.com/websockets/v3?app_id=1089",
        "wss://ws.binaryws.com/websockets/v3?app_id=16929",
        "wss://ws.binaryws.com/websockets/v3"
    ]
    
    for ws_url in urls_to_try:
        try:
            logger.info(f"Testing: {ws_url}")
            
            # Connect to WebSocket
            websocket = await websockets.connect(ws_url)
            logger.info("✅ Connected!")
            
            # Send authorization immediately
            auth_request = {
                "authorize": token,
                "req_id": 1
            }
            
            await websocket.send(json.dumps(auth_request))
            logger.info("📤 Auth request sent")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=10)
            data = json.loads(response)
            
            logger.info(f"📨 Response: {data}")
            
            if 'authorize' in data:
                logger.info("🎉 SUCCESS!")
                logger.info(f"Account: {data['authorize'].get('loginid', 'N/A')}")
                logger.info(f"Currency: {data['authorize'].get('currency', 'N/A')}")
                logger.info(f"Balance: {data['authorize'].get('balance', 'N/A')}")
                logger.info(f"Is Demo: {'demo' in data['authorize'].get('loginid', '').lower()}")
                
                await websocket.close()
                return ws_url, True
                
            elif 'error' in data:
                logger.error(f"❌ Auth error: {data['error']}")
                
            await websocket.close()
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            continue
    
    return None, False

async def test_app_registration():
    """Try registering the app first"""
    try:
        ws_url = "wss://ws.binaryws.com/websockets/v3"
        websocket = await websockets.connect(ws_url)
        
        # Register app
        app_request = {
            "app_register": 1,
            "name": "Deriv AI Bot",
            "scopes": ["read", "trade", "trading_information"],
            "req_id": 1
        }
        
        await websocket.send(json.dumps(app_request))
        response = await asyncio.wait_for(websocket.recv(), timeout=10)
        data = json.loads(response)
        
        logger.info(f"App registration: {data}")
        
        if 'app_register' in data:
            app_id = data['app_register']['app_id']
            logger.info(f"✅ App registered with ID: {app_id}")
            return app_id
            
        await websocket.close()
        
    except Exception as e:
        logger.error(f"App registration failed: {e}")
        
    return None

async def main():
    print("🔧 Testing Deriv API with Different Methods...")
    print("=" * 60)
    
    # Try app registration first
    print("\n1️⃣ Trying app registration...")
    app_id = await test_app_registration()
    
    # Try with token
    print("\n2️⃣ Testing with token...")
    working_url, success = await test_with_token_in_url()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"Working URL: {working_url}")
        print("✅ Ready to start the bot!")
        
        # Update config with working URL
        return working_url
    else:
        print("\n❌ All methods failed")
        print("🔧 Troubleshooting steps:")
        print("1. Verify token at: https://app.deriv.com/account/api-token")
        print("2. Check token scopes: Read, Trade, Trading Information")
        print("3. Ensure token is active and not expired")
        print("4. Try generating a new token")
        
    return None

if __name__ == "__main__":
    result = asyncio.run(main())
