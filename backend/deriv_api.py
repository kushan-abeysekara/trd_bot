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
        self.active_contracts = {}  # Track active contracts for outcome monitoring
        self.contract_callbacks = {}  # Callbacks for contract outcomes
        
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
            msg_type = data.get("msg_type")
            
            if msg_type == "authorize":
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
                    
            elif msg_type == "balance":
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
                
            elif msg_type == "buy":
                # Handle buy response
                req_id = data.get("req_id")
                if req_id in self.callbacks:
                    self.callbacks[req_id](data)
                    del self.callbacks[req_id]
                    
            elif msg_type == "proposal":
                # Handle proposal response
                req_id = data.get("req_id")
                if req_id in self.callbacks:
                    self.callbacks[req_id](data)
                    del self.callbacks[req_id]
                    
            elif msg_type == "proposal_open_contract":
                # Handle contract updates (REAL CONTRACT OUTCOMES)
                self._handle_contract_update(data)
                
            elif msg_type == "transaction":
                # Handle transaction updates (includes contract settlements)
                self._handle_transaction_update(data)
                
        except json.JSONDecodeError:
            print(f"Failed to parse message: {message}")
    
    def _handle_contract_update(self, data):
        """Handle real contract outcome updates from Deriv"""
        try:
            contract_data = data.get("proposal_open_contract", {})
            contract_id = contract_data.get("contract_id")
            contract_status = contract_data.get("status")
            
            if contract_id and contract_id in self.active_contracts:
                contract_info = self.active_contracts[contract_id]
                
                # Check if contract is settled
                if contract_status in ["sold", "won", "lost"]:
                    # Get actual profit/loss
                    profit = float(contract_data.get("profit", 0))
                    payout = float(contract_data.get("payout", 0))
                    buy_price = float(contract_data.get("buy_price", 0))
                    
                    # Determine actual result
                    is_win = profit > 0
                    actual_profit_loss = profit
                    
                    print(f"🎯 REAL CONTRACT OUTCOME - ID: {contract_id}")
                    print(f"   Status: {contract_status}")
                    print(f"   Result: {'WIN' if is_win else 'LOSS'}")
                    print(f"   Buy Price: ${buy_price:.2f}")
                    print(f"   Payout: ${payout:.2f}")
                    print(f"   Profit/Loss: ${actual_profit_loss:.2f}")
                    
                    # Update contract info with real result
                    contract_info.update({
                        'real_result': 'win' if is_win else 'loss',
                        'real_profit_loss': actual_profit_loss,
                        'real_payout': payout,
                        'settlement_time': datetime.now().isoformat(),
                        'deriv_status': contract_status
                    })
                    
                    # Call contract callback if set
                    if contract_id in self.contract_callbacks:
                        try:
                            self.contract_callbacks[contract_id](contract_info)
                            del self.contract_callbacks[contract_id]
                        except Exception as e:
                            print(f"Error in contract callback: {e}")
                    
                    # Remove from active contracts
                    del self.active_contracts[contract_id]
                    
        except Exception as e:
            print(f"Error handling contract update: {e}")
    
    def _handle_transaction_update(self, data):
        """Handle transaction updates including contract settlements"""
        try:
            transaction = data.get("transaction", {})
            action = transaction.get("action")
            contract_id = transaction.get("contract_id")
            
            if action == "sell" and contract_id:
                # Contract was settled
                amount = float(transaction.get("amount", 0))
                
                if contract_id in self.active_contracts:
                    contract_info = self.active_contracts[contract_id]
                    buy_price = contract_info.get('buy_price', 0)
                    
                    # Calculate actual profit/loss
                    actual_profit_loss = amount - buy_price
                    is_win = actual_profit_loss > 0
                    
                    print(f"💰 TRANSACTION SETTLEMENT - Contract: {contract_id}")
                    print(f"   Settlement Amount: ${amount:.2f}")
                    print(f"   Original Stake: ${buy_price:.2f}")
                    print(f"   Actual P&L: ${actual_profit_loss:.2f}")
                    print(f"   Result: {'WIN' if is_win else 'LOSS'}")
                    
                    # Update with transaction data
                    contract_info.update({
                        'transaction_result': 'win' if is_win else 'loss',
                        'transaction_profit_loss': actual_profit_loss,
                        'settlement_amount': amount,
                        'transaction_time': datetime.now().isoformat()
                    })
                    
        except Exception as e:
            print(f"Error handling transaction update: {e}")
            
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
    
    def buy_contract(self, proposal_id: str, price: float, callback, contract_info=None):
        """Buy a contract and track it for real outcome monitoring"""
        if not self.is_connected:
            callback({"error": {"message": "Not connected to API"}})
            return
        
        # Wrap the original callback to track contracts and refresh balance
        def enhanced_callback(response):
            # Validate the trade result structure
            if not response.get("error") and "buy" in response:
                # Store contract details for real outcome tracking
                buy_data = response["buy"]
                contract_id = buy_data.get("contract_id")
                buy_price = float(buy_data.get("buy_price", 0))
                
                if contract_id:
                    # Store contract info for outcome tracking
                    self.active_contracts[contract_id] = {
                        'contract_id': contract_id,
                        'buy_price': buy_price,
                        'proposal_id': proposal_id,
                        'buy_time': datetime.now().isoformat(),
                        'contract_info': contract_info or {}
                    }
                    
                    # Subscribe to contract updates
                    self._subscribe_to_contract(contract_id)
                    
                    print(f"🧾 Contract tracked for real outcome - ID: {contract_id}, Buy price: ${buy_price}")
            
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
    
    def _subscribe_to_contract(self, contract_id: str):
        """Subscribe to contract updates for real outcome tracking"""
        if not self.is_connected:
            return
            
        try:
            subscribe_request = {
                "proposal_open_contract": 1,
                "contract_id": contract_id,
                "subscribe": 1,
                "req_id": self.req_id
            }
            
            self.ws.send(json.dumps(subscribe_request))
            print(f"📡 Subscribed to contract updates: {contract_id}")
            self.req_id += 1
            
        except Exception as e:
            print(f"Error subscribing to contract: {e}")
    
    def set_contract_callback(self, contract_id: str, callback):
        """Set callback for specific contract outcome"""
        self.contract_callbacks[contract_id] = callback
    
    def get_active_contracts(self):
        """Get list of active contracts being tracked"""
        return list(self.active_contracts.keys())
    
    def get_balance_value(self):
        """Get current balance value with validation"""
        # If balance is zero and we're connected, try to refresh
        if self.balance <= 0 and self.is_connected:
            print("Balance is zero, attempting to refresh")
            self._get_balance()
            # Small delay to allow potential response to arrive
            time.sleep(0.5)
        return max(0.0, self.balance)  # Ensure non-negative balance
        
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
            self.active_contracts.clear()
            self.contract_callbacks.clear()
    
    def update_balance(self, profit_loss: float):
        """Update balance with profit/loss from a trade"""
        if self.balance > 0:
            old_balance = self.balance
            self.balance += profit_loss
            print(f"Balance updated: {'+' if profit_loss >= 0 else ''}${profit_loss:.2f} -> ${self.balance:.2f}")
            
            # Trigger balance callback if set
            if self.balance_callback:
                try:
                    self.balance_callback(self.balance)
                except Exception as e:
                    print(f"Error in balance callback: {e}")
        else:
            print(f"Cannot update balance: current balance is {self.balance}")
