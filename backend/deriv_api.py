import json
import asyncio
import websocket
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional


class DerivAPI:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.ws = None
        self.is_connected = False
        self.balance = 0.0
        self.req_id = 1
        self.callbacks = {}
        self.connection_callback = None
        self.balance_callback = None  # Callback for balance updates
        
    def connect(self, callback=None):
        """Connect to Deriv WebSocket API"""
        self.connection_callback = callback
        websocket.enableTrace(False)
        self.ws = websocket.WebSocketApp(
            "wss://ws.binaryws.com/websockets/v3?app_id=1089",
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
    def _on_open(self, ws):
        """Called when WebSocket connection is opened"""
        print("Connected to Deriv API")
        self.is_connected = True
        
        # Authorize with API token
        auth_request = {
            "authorize": self.api_token,
            "req_id": self.req_id
        }
        self.ws.send(json.dumps(auth_request))
        self.req_id += 1
        
        if self.connection_callback:
            self.connection_callback(True)
            
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get("msg_type") == "authorize":
                if data.get("error"):
                    print(f"Authorization failed: {data['error']['message']}")
                    if self.connection_callback:
                        self.connection_callback(False, data['error']['message'])
                else:
                    print("Authorization successful")
                    # Make multiple balance requests for reliability
                    self._get_balance()
                    # Schedule another balance request after a short delay
                    threading.Timer(2.0, self._get_balance).start()
                    
            elif data.get("msg_type") == "balance":
                old_balance = self.balance
                # Extract balance more safely with better error checking
                if "balance" in data and data["balance"] is not None:
                    balance_data = data["balance"]
                    new_balance = float(balance_data.get("balance", 0))
                    
                    # Only update if we get a valid non-zero balance
                    if new_balance > 0:
                        self.balance = new_balance
                        print(f"Balance updated: ${old_balance:.2f} -> ${self.balance:.2f}")
                        
                        # Trigger balance callback if set
                        if self.balance_callback:
                            try:
                                self.balance_callback(self.balance)
                            except Exception as e:
                                print(f"Error in balance callback: {e}")
                    else:
                        print(f"Received zero balance update, ignoring")
                else:
                    print(f"Received invalid balance data: {data}")
                
            elif data.get("msg_type") == "buy":
                # Handle buy response
                req_id = data.get("req_id")
                if req_id in self.callbacks:
                    self.callbacks[req_id](data)
                    del self.callbacks[req_id]
                    
            elif data.get("msg_type") == "proposal":
                # Handle proposal response
                req_id = data.get("req_id")
                if req_id in self.callbacks:
                    self.callbacks[req_id](data)
                    del self.callbacks[req_id]
                    
        except json.JSONDecodeError:
            print(f"Failed to parse message: {message}")
            
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
        self.is_connected = False
        
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print("WebSocket connection closed")
        self.is_connected = False
        
    def _get_balance(self):
        """Get account balance"""
        if not self.is_connected:
            print("Cannot get balance: not connected")
            return
            
        try:
            balance_request = {
                "balance": 1,
                "subscribe": 1,  # Subscribe to balance updates
                "req_id": self.req_id
            }
            self.ws.send(json.dumps(balance_request))
            print("Balance request sent")
            self.req_id += 1
        except Exception as e:
            print(f"Error requesting balance: {e}")
    
    def set_balance_callback(self, callback):
        """Set callback for balance updates"""
        self.balance_callback = callback
        
    def refresh_balance(self):
        """Request fresh balance data from server"""
        if self.is_connected:
            try:
                self._get_balance()
            except Exception as e:
                print(f"Error refreshing balance: {e}")
                
    def get_proposal(self, contract_type: str, duration: int, amount: float, callback):
        """Get contract proposal for tick-based trades"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
            
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": duration,
            "duration_unit": "t",  # ticks
            "symbol": "R_100",  # Volatility 100 Index
            "req_id": self.req_id
        }
        
        print(f"Requesting proposal for {contract_type} trade, {duration} tick duration, ${amount} stake")
        self.callbacks[self.req_id] = callback
        try:
            self.ws.send(json.dumps(proposal_request))
            print(f"Proposal request sent with req_id: {self.req_id}")
        except Exception as e:
            print(f"Error sending proposal request: {e}")
            callback({"error": {"message": f"Failed to send request: {str(e)}"}})
            
        self.req_id += 1
    
    def get_proposal_ticks(self, contract_type: str, ticks: int, amount: float, callback):
        """Get contract proposal specifically for tick-based trades"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
            
        proposal_request = {
            "proposal": 1,
            "amount": amount,
            "basis": "stake",
            "contract_type": contract_type,
            "currency": "USD",
            "duration": ticks,
            "duration_unit": "t",  # ticks
            "symbol": "R_100",  # Volatility 100 Index
            "req_id": self.req_id
        }
        
        print(f"Requesting proposal for {contract_type} trade, {ticks} tick duration, ${amount} stake")
        self.callbacks[self.req_id] = callback
        try:
            self.ws.send(json.dumps(proposal_request))
            print(f"Proposal request sent with req_id: {self.req_id}")
        except Exception as e:
            print(f"Error sending proposal request: {e}")
            callback({"error": {"message": f"Failed to send request: {str(e)}"}})
            
        self.req_id += 1
    
    def buy_contract(self, proposal_id: str, price: float, callback):
        """Buy a contract"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
            
        # Wrap the original callback to refresh balance after trade and validate contract
        def enhanced_callback(response):
            # Validate the trade result structure
            if not response.get("error") and "buy" in response:
                # Store contract details for later validation
                contract_info = response["buy"]
                contract_id = contract_info.get("contract_id")
                buy_price = float(contract_info.get("buy_price", 0))
                
                print(f"🧾 Contract validated - ID: {contract_id}, Buy price: ${buy_price}")
                # Store the contract details for later outcome validation
            
            # Call the original callback
            callback(response)
            
            # Refresh balance after trade completion (whether success or error)
            if not response.get("error"):
                # Immediately refresh balance to get updated value
                threading.Timer(1.0, self.refresh_balance).start()
                # Also refresh again after a short delay to ensure we get the final balance
                threading.Timer(3.0, self.refresh_balance).start()
            
        buy_request = {
            "buy": proposal_id,
            "price": price,
            "req_id": self.req_id
        }
        
        print(f"Buying contract with proposal ID: {proposal_id}, price: ${price}")
        self.callbacks[self.req_id] = enhanced_callback
        try:
            self.ws.send(json.dumps(buy_request))
            print(f"Buy request sent with req_id: {self.req_id}")
        except Exception as e:
            print(f"Error sending buy request: {e}")
            callback({"error": {"message": f"Failed to send buy request: {str(e)}"}})
            
        self.req_id += 1
    
    def update_balance(self, profit_loss: float):
        """Update balance with profit/loss from trade"""
        old_balance = self.balance
        self.balance += profit_loss
        print(f"Balance updated: ${profit_loss:+.2f} -> New Balance: ${self.balance:.2f}")
        
        # After updating balance locally, request fresh balance from server
        threading.Timer(1.0, self.refresh_balance).start()
        
        # Emit balance update
        if self.balance_callback:
            try:
                self.balance_callback(self.balance)
            except Exception as e:
                print(f"Error in balance callback: {e}")
        
    def get_balance_value(self):
        """Get current balance"""
        # If balance is zero and we're connected, try to refresh
        if self.balance <= 0 and self.is_connected:
            print("Balance is zero, attempting to refresh")
            self._get_balance()
            # Small delay to allow potential response to arrive
            time.sleep(0.5)
        return self.balance
        
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
