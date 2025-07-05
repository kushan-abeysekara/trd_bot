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
                    self._get_balance()
                    
            elif data.get("msg_type") == "balance":
                old_balance = self.balance
                self.balance = float(data.get("balance", {}).get("balance", 0))
                print(f"Balance updated: ${old_balance:.2f} -> ${self.balance:.2f}")
                
                # Trigger balance callback if set
                if self.balance_callback:
                    self.balance_callback(self.balance)
                
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
            
            elif data.get("msg_type") == "proposal_open_contract":
                # Handle contract monitoring updates
                req_id = data.get("req_id")
                if req_id in self.callbacks:
                    # Only delete callback when contract is finished
                    contract = data.get("proposal_open_contract", {})
                    is_finished = contract.get("status") != "open"
                    
                    self.callbacks[req_id](data)
                    
                    if is_finished:
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
        balance_request = {
            "balance": 1,
            "req_id": self.req_id
        }
        self.ws.send(json.dumps(balance_request))
        self.req_id += 1
        
    def set_balance_callback(self, callback):
        """Set callback for balance updates"""
        self.balance_callback = callback
        
    def refresh_balance(self):
        """Request fresh balance data from server"""
        if self.is_connected:
            self._get_balance()
        
    def get_proposal(self, contract_type: str, duration: int, amount: float, callback):
        """Get contract proposal"""
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
            "duration_unit": "s",  # seconds (changed from ticks)
            "symbol": "R_100",  # Volatility 100 Index
            "req_id": self.req_id
        }
        
        self.callbacks[self.req_id] = callback
        self.ws.send(json.dumps(proposal_request))
        self.req_id += 1
        
    def buy_contract(self, proposal_id: str, price: float, callback):
        """Buy a contract"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
            
        # Wrap the original callback to refresh balance after trade
        def enhanced_callback(response):
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
        
        self.callbacks[self.req_id] = enhanced_callback
        self.ws.send(json.dumps(buy_request))
        self.req_id += 1
        
    def update_balance(self, profit_loss: float):
        """Update balance with profit/loss from trade"""
        self.balance += profit_loss
        print(f"Balance updated: ${profit_loss:+.2f} -> New Balance: ${self.balance:.2f}")
        
    def get_balance_value(self):
        """Get current balance"""
        return self.balance
        
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
            
    def monitor_contract(self, contract_id, callback):
        """Monitor a specific contract for real result verification"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
            
        proposal_request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
            "req_id": self.req_id
        }
        
        self.callbacks[self.req_id] = callback
        self.ws.send(json.dumps(proposal_request))
        self.req_id += 1
        
    def subscribe_to_contract(self, contract_id, callback):
        """Subscribe to contract updates for a specific contract ID"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
            
        proposal_open_request = {
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
            "req_id": self.req_id
        }
        
        self.callbacks[self.req_id] = callback
        self.ws.send(json.dumps(proposal_open_request))
        
        # Return the subscription ID so it can be used to unsubscribe later
        subscription_id = self.req_id
        self.req_id += 1
        return subscription_id
        
    def unsubscribe(self, subscription_id):
        """Unsubscribe from a previous subscription"""
        if not self.is_connected:
            return
            
        forget_request = {
            "forget": subscription_id,
            "req_id": self.req_id
        }
        
        self.ws.send(json.dumps(forget_request))
        self.req_id += 1
        
        # Remove callback if it exists
        if subscription_id in self.callbacks:
            del self.callbacks[subscription_id]
