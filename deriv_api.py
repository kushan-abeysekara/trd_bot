"""
Deriv API WebSocket client for trading operations
"""
import websockets
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import config

logger = logging.getLogger(__name__)

class DerivAPI:
    def __init__(self, token: str = None):
        self.token = token or config.DERIV_TOKEN
        self.ws_url = config.DERIV_WS_URL
        self.websocket = None
        self.is_connected = False
        self.request_id = 0
        self.callbacks = {}
        self.subscriptions = {}
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3  # Reduced reconnection attempts
        self.reconnect_delay = 10  # Increased initial delay
        self.ping_interval = 60  # Increased ping interval
        self.last_ping = None
        self.connection_lost_callback = None
        self._connection_lock = asyncio.Lock()  # Prevent concurrent connections
        self._listen_task = None
        self._ping_task = None
        self._is_reconnecting = False
        
    async def connect(self):
        """Connect to Deriv WebSocket API with improved stability"""
        # Prevent concurrent connection attempts
        async with self._connection_lock:
            if self.is_connected and self.websocket:
                logger.info("Already connected to Deriv API")
                return True
                
            if self._is_reconnecting:
                logger.info("Reconnection already in progress")
                return False
                
            self._is_reconnecting = True
            self.reconnect_attempts = 0
            
            try:
                while self.reconnect_attempts < self.max_reconnect_attempts:
                    try:
                        logger.info(f"🔄 Connecting to Deriv WebSocket: {self.ws_url} (attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})")
                        
                        # Clean up any existing connection
                        await self._cleanup_connection()
                        
                        # Connect with extended timeout and better settings
                        self.websocket = await asyncio.wait_for(
                            websockets.connect(
                                self.ws_url,
                                ping_interval=None,  # Disable automatic ping
                                ping_timeout=None,   # Disable ping timeout
                                close_timeout=15,
                                max_size=10**6,      # 1MB max message size
                                compression=None,    # Disable compression
                                extra_headers={
                                    "User-Agent": "DerivAI-TradingBot/2.0",
                                    "Origin": "https://app.deriv.com"
                                }
                            ),
                            timeout=45  # Extended connection timeout
                        )
                        
                        # Test connection immediately
                        if not await self._test_connection():
                            logger.error("❌ Connection test failed immediately after connecting")
                            await self._cleanup_connection()
                            raise Exception("Connection test failed")
                        
                        self.is_connected = True
                        self.reconnect_attempts = 0
                        self.reconnect_delay = 10  # Reset delay
                        logger.info("✅ Successfully connected to Deriv WebSocket API")
                        
                        # Start background tasks
                        self._listen_task = asyncio.create_task(self._listen())
                        self._ping_task = asyncio.create_task(self._ping_loop())
                        
                        # Authorize with token if provided
                        if self.token:
                            try:
                                auth_success = await self.authorize()
                                if not auth_success:
                                    logger.warning("⚠️ Authorization failed, continuing in read-only mode")
                            except Exception as e:
                                logger.warning(f"⚠️ Authorization error: {e}, continuing in read-only mode")
                        else:
                            logger.info("ℹ️ No token provided, running in read-only mode")
                            
                        return True
                            
                    except asyncio.TimeoutError:
                        self.reconnect_attempts += 1
                        logger.error(f"⏱️ Connection timeout (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                        
                    except Exception as e:
                        self.reconnect_attempts += 1
                        logger.error(f"❌ Failed to connect (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}): {e}")
                        
                    # Wait before retry with exponential backoff
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        wait_time = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
                        wait_time = min(wait_time, 120)  # Max 2 minutes
                        logger.info(f"🔄 Retrying connection in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error("❌ Max reconnection attempts reached")
                        self.is_connected = False
                        return False
                
                return False
                
            finally:
                self._is_reconnecting = False
        
    async def _test_connection(self):
        """Test connection with ping - improved version"""
        try:
            if not self.websocket:
                return False
                
            ping_request = {"ping": 1, "req_id": 999999}
            await asyncio.wait_for(
                self.websocket.send(json.dumps(ping_request)),
                timeout=10
            )
            
            # Wait for pong response
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=15
            )
            
            data = json.loads(response)
            if 'pong' in data:
                logger.info("✅ Connection test successful (ping/pong)")
                return True
            else:
                logger.warning(f"⚠️ Unexpected ping response: {data}")
                return True  # Some responses might be valid but different format
                
        except asyncio.TimeoutError:
            logger.error("❌ Connection test timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
            
    async def _cleanup_connection(self):
        """Clean up existing connection and tasks"""
        # Cancel background tasks
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        # Close websocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            finally:
                self.websocket = None
        
        self.is_connected = False
        self._listen_task = None
        self._ping_task = None
        
    async def disconnect(self):
        """Disconnect from WebSocket"""
        logger.info("🔌 Disconnecting from Deriv WebSocket API...")
        await self._cleanup_connection()
        logger.info("✅ Disconnected from Deriv WebSocket API")
            
    async def _listen(self):
        """Listen for incoming messages with improved error handling and stability"""
        logger.info("🎧 Starting message listener...")
        
        try:
            while self.is_connected and self.websocket:
                try:
                    # Wait for message with longer timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=120  # 2 minute timeout - more generous
                    )
                    
                    if message:
                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ Invalid JSON received: {e}")
                            continue
                        
                except asyncio.TimeoutError:
                    logger.debug("📡 No message received in timeout period, checking connection...")
                    # Only test connection if we haven't received messages for a while
                    if not await self._test_connection():
                        logger.error("❌ Connection test failed during timeout")
                        break
                    else:
                        logger.debug("✅ Connection still alive")
                        continue
                        
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"🔌 WebSocket connection closed: {e}")
                    break
                    
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.warning(f"🔌 WebSocket connection closed unexpectedly: {e}")
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Error receiving message: {e}")
                    # Test if connection is still valid before breaking
                    if not self.websocket or self.websocket.closed:
                        logger.error("WebSocket is closed, stopping listener")
                        break
                    # Continue for recoverable errors
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("🔴 Message listener was cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Critical error in message listener: {e}")
        finally:
            logger.warning("🔴 Message listener stopped")
            self.is_connected = False
            
            # Don't trigger automatic reconnection from listener
            # Let the external monitor handle it
            if self.connection_lost_callback:
                try:
                    asyncio.create_task(self.connection_lost_callback())
                except Exception as e:
                    logger.error(f"Error in connection lost callback: {e}")
            
    async def _ping_loop(self):
        """Send periodic pings to keep connection alive - improved version"""
        logger.info("💓 Starting ping loop...")
        
        try:
            while self.is_connected and self.websocket:
                try:
                    # Wait for ping interval
                    await asyncio.sleep(self.ping_interval)
                    
                    if not self.is_connected or not self.websocket:
                        break
                    
                    # Send ping
                    ping_request = {"ping": 1, "req_id": self._get_request_id()}
                    await asyncio.wait_for(
                        self.websocket.send(json.dumps(ping_request)),
                        timeout=10
                    )
                    
                    self.last_ping = datetime.now()
                    logger.debug("💓 Ping sent")
                    
                except asyncio.TimeoutError:
                    logger.error("⏱️ Ping timeout - connection may be lost")
                    self.is_connected = False
                    break
                    
                except asyncio.CancelledError:
                    logger.info("💓 Ping loop cancelled")
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Error sending ping: {e}")
                    if not self.websocket or self.websocket.closed:
                        self.is_connected = False
                        break
                    # Continue for recoverable errors
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("💓 Ping loop was cancelled")
            raise
        except Exception as e:
            logger.error(f"❌ Critical error in ping loop: {e}")
        finally:
            logger.warning("🔴 Ping loop stopped")
                
    async def _attempt_reconnection(self):
        """Attempt to reconnect after connection loss - REMOVED AUTOMATIC RECONNECTION"""
        # Don't automatically reconnect from API level
        # Let the dashboard/external monitor handle reconnections
        logger.info("Connection lost - external monitor will handle reconnection")
        pass
            
    async def _handle_message(self, data: Dict[Any, Any]):
        """Handle incoming WebSocket messages"""
        req_id = data.get('req_id')
        msg_type = data.get('msg_type')
        
        # Handle subscription updates
        if 'subscription' in data:
            subscription_id = data['subscription']['id']
            if subscription_id in self.subscriptions:
                callback = self.subscriptions[subscription_id]
                if callback:
                    await callback(data)
        
        # Handle request responses
        elif req_id and req_id in self.callbacks:
            callback = self.callbacks[req_id]
            if callback:
                await callback(data)
                del self.callbacks[req_id]
                
    def _get_request_id(self) -> int:
        """Generate unique request ID"""
        self.request_id += 1
        return self.request_id
        
    async def _send_request(self, request: Dict[Any, Any], callback=None) -> int:
        """Send request to WebSocket"""
        req_id = self._get_request_id()
        request['req_id'] = req_id
        
        if callback:
            self.callbacks[req_id] = callback
            
        await self.websocket.send(json.dumps(request))
        return req_id
        
    async def authorize(self):
        """Authorize with Deriv API"""
        if not self.token:
            logger.warning("No token provided, running in read-only mode")
            return True  # Allow read-only mode
            
        request = {
            "authorize": self.token
        }
        
        result = await self._send_request_sync(request)
        if 'error' in result:
            logger.warning(f"Authorization failed: {result['error']['message']}")
            logger.info("Continuing in read-only mode (market data only)")
            return True  # Continue without authorization
            
        logger.info("Successfully authorized with Deriv API")
        return True
        
    async def _send_request_sync(self, request: Dict[Any, Any]) -> Dict[Any, Any]:
        """Send request and wait for response with timeout"""
        if not self.is_connected or not self.websocket:
            logger.error("❌ Cannot send request - not connected")
            return {"error": {"message": "Not connected to API"}}
            
        response = {}
        response_received = asyncio.Event()
        
        async def callback(data):
            nonlocal response
            response = data
            response_received.set()
            
        try:
            # Send request
            req_id = await self._send_request(request, callback)
            
            # Wait for response with extended timeout
            try:
                await asyncio.wait_for(response_received.wait(), timeout=30.0)  # Extended timeout
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Request timeout for req_id {req_id}")
                # Clean up callback
                if req_id in self.callbacks:
                    del self.callbacks[req_id]
                return {"error": {"message": "Request timeout"}}
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error sending request: {e}")
            return {"error": {"message": f"Request failed: {str(e)}"}}
        
    async def _send_request(self, request: Dict[Any, Any], callback=None) -> int:
        """Send request to WebSocket with improved error handling"""
        if not self.is_connected or not self.websocket:
            raise Exception("Not connected to WebSocket")
            
        req_id = self._get_request_id()
        request['req_id'] = req_id
        
        if callback:
            self.callbacks[req_id] = callback
            
        try:
            await asyncio.wait_for(
                self.websocket.send(json.dumps(request)),
                timeout=5.0
            )
            logger.debug(f"📤 Request sent: {req_id}")
            return req_id
        except Exception as e:
            # Clean up callback on error
            if req_id in self.callbacks:
                del self.callbacks[req_id]
            raise Exception(f"Failed to send request: {e}")
        
    async def get_account_info(self) -> Dict[Any, Any]:
        """Get account information"""
        request = {"get_account_status": 1}
        return await self._send_request_sync(request)
        
    async def get_balance(self) -> Dict[Any, Any]:
        """Get account balance"""
        request = {"balance": 1, "subscribe": 1}
        return await self._send_request_sync(request)
        
    async def get_ticks(self, symbol: str, callback=None) -> Dict[Any, Any]:
        """Subscribe to tick data"""
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        
        if callback:
            result = await self._send_request_sync(request)
            if 'subscription' in result:
                self.subscriptions[result['subscription']['id']] = callback
                
        return await self._send_request_sync(request)
        
    async def get_candles(self, symbol: str, timeframe: str = "60", count: int = 100) -> Dict[Any, Any]:
        """Get candlestick data"""
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "granularity": int(timeframe),
            "style": "candles"
        }
        
        return await self._send_request_sync(request)
        
    async def buy_contract(self, 
                          symbol: str, 
                          contract_type: str, 
                          amount: float, 
                          duration: int = 5, 
                          duration_unit: str = "t",
                          basis: str = "stake") -> Dict[Any, Any]:
        """Place a buy order"""
        request = {
            "buy": 1,
            "price": amount,
            "parameters": {
                "contract_type": contract_type.upper(),
                "symbol": symbol,
                "amount": amount,
                "duration": duration,
                "duration_unit": duration_unit,
                "basis": basis
            }
        }
        
        result = await self._send_request_sync(request)
        logger.info(f"Buy order placed: {result}")
        return result
        
    async def sell_contract(self, contract_id: str, price: float) -> Dict[Any, Any]:
        """Sell a contract"""
        request = {
            "sell": contract_id,
            "price": price
        }
        
        result = await self._send_request_sync(request)
        logger.info(f"Sell order placed: {result}")
        return result
        
    async def get_contract_info(self, contract_id: str) -> Dict[Any, Any]:
        """Get contract information"""
        request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1
        }
        
        return await self._send_request_sync(request)
        
    async def get_active_symbols(self) -> Dict[Any, Any]:
        """Get list of active trading symbols"""
        request = {
            "active_symbols": "brief",
            "product_type": "basic"
        }
        
        return await self._send_request_sync(request)
        
    async def get_trading_times(self, symbol: str) -> Dict[Any, Any]:
        """Get trading times for a symbol"""
        request = {
            "trading_times": symbol
        }
        
        return await self._send_request_sync(request)
        
    async def get_price_proposal(self, 
                                symbol: str, 
                                contract_type: str, 
                                amount: float,
                                duration: int = 5,
                                duration_unit: str = "t",
                                basis: str = "stake") -> Dict[Any, Any]:
        """Get price proposal for a contract"""
        request = {
            "proposal": 1,
            "amount": amount,
            "basis": basis,
            "contract_type": contract_type.upper(),
            "currency": "USD",
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }
        
        return await self._send_request_sync(request)

# Example usage
if __name__ == "__main__":
    async def test_api():
        api = DerivAPI()
        await api.connect()
        
        # Get account info
        account_info = await api.get_account_info()
        print("Account Info:", account_info)
        
        # Get balance
        balance = await api.get_balance()
        print("Balance:", balance)
        
        # Get tick data
        ticks = await api.get_ticks("R_50")
        print("Ticks:", ticks)
        
        await api.disconnect()
        
    asyncio.run(test_api())
