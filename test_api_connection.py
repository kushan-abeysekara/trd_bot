#!/usr/bin/env python3
"""
Test Deriv API Connection
"""
import asyncio
import websockets
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_deriv_connection():
    """Test connection to Deriv API"""
    
    # Your API credentials (Demo Account)
    token = "6D0lReGW3insnlx"  # Your demo token
    # Try the frontend WebSocket endpoint
    ws_url = "wss://ws.binaryws.com/websockets/v3?app_id=1089"
    
    try:
        logger.info(f"Testing connection to: {ws_url}")
        logger.info(f"Using token: {token[:10]}...")
        
        # Connect to WebSocket
        websocket = await websockets.connect(ws_url)
        logger.info("✅ Connected to Deriv WebSocket")
        
        # Test authorization
        auth_request = {
            "authorize": token,
            "req_id": 1
        }
        
        await websocket.send(json.dumps(auth_request))
        logger.info("📤 Authorization request sent")
        
        # Wait for response
        response = await websocket.recv()
        data = json.loads(response)
        
        logger.info(f"📨 Response received: {data}")
        
        if 'error' in data:
            logger.error(f"❌ Authorization failed: {data['error']}")
            return False
        elif 'authorize' in data:
            logger.info("✅ Authorization successful!")
            logger.info(f"Account ID: {data['authorize'].get('loginid', 'N/A')}")
            logger.info(f"Currency: {data['authorize'].get('currency', 'N/A')}")
            logger.info(f"Balance: {data['authorize'].get('balance', 'N/A')}")
            return True
        else:
            logger.warning(f"⚠️  Unexpected response: {data}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Connection test failed: {e}")
        return False
    finally:
        if 'websocket' in locals():
            await websocket.close()
            logger.info("🔌 WebSocket connection closed")

async def test_app_registration():
    """Test app registration if needed"""
    ws_url = "wss://ws.derivws.com/websockets/v3"
    
    try:
        websocket = await websockets.connect(ws_url)
        
        # Register app
        app_request = {
            "app_register": 1,
            "name": "Deriv AI Trading Bot",
            "scopes": ["read", "trade", "payments", "admin"],
            "req_id": 2
        }
        
        await websocket.send(json.dumps(app_request))
        response = await websocket.recv()
        data = json.loads(response)
        
        logger.info(f"App registration response: {data}")
        
    except Exception as e:
        logger.error(f"App registration test failed: {e}")
    finally:
        if 'websocket' in locals():
            await websocket.close()

async def main():
    """Main test function"""
    print("🔧 Testing Deriv API Connection...")
    print("=" * 50)
    
    # Test basic connection and authorization
    success = await test_deriv_connection()
    
    if not success:
        print("\n🔧 If authorization failed, you may need to:")
        print("1. Check if your token is still valid")
        print("2. Verify token permissions (read, trade)")
        print("3. Check if token is for the correct environment (demo/live)")
        print("4. Visit: https://app.deriv.com/account/api-token")
        
        print("\n🔧 Testing app registration...")
        await test_app_registration()
    
    print("\n" + "=" * 50)
    print("✅ Connection test completed!" if success else "❌ Connection test failed!")

if __name__ == "__main__":
    asyncio.run(main())
