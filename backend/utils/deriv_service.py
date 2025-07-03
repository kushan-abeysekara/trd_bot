import websocket
import json
import threading
import time
from datetime import datetime
import requests
import ssl

class DerivService:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.app_id = 1089  # Default app ID - users can use their own
        # Include app_id in WebSocket URL
        self.ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
        
    def validate_token(self, api_token):
        """Validate API token with Deriv"""
        try:
            # Create WebSocket connection for validation with SSL context
            ws = websocket.create_connection(
                self.ws_url,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                timeout=10
            )
            
            # Send authorize request
            auth_request = {
                "authorize": api_token,
                "req_id": 1
            }
            
            ws.send(json.dumps(auth_request))
            
            # Set timeout for receiving response
            ws.settimeout(10)
            response = json.loads(ws.recv())
            
            ws.close()
            
            if 'error' in response:
                return {
                    'valid': False,
                    'message': response['error']['message'],
                    'code': response['error'].get('code', 'UNKNOWN')
                }
            
            if 'authorize' in response:
                return {
                    'valid': True,
                    'message': 'Token is valid',
                    'data': response['authorize']
                }
            
            return {
                'valid': False,
                'message': 'Invalid response from Deriv'
            }
            
        except websocket.WebSocketTimeoutException:
            return {
                'valid': False,
                'message': 'Connection timeout. Please check your internet connection.'
            }
        except websocket.WebSocketConnectionClosedException:
            return {
                'valid': False,
                'message': 'Connection closed unexpectedly.'
            }
        except Exception as e:
            print(f"Token validation error: {str(e)}")
            error_message = str(e)
            
            # Handle specific error cases
            if "InvalidAppID" in error_message:
                return {
                    'valid': False,
                    'message': 'Invalid application ID. Please contact support.'
                }
            elif "401" in error_message or "Unauthorized" in error_message:
                return {
                    'valid': False,
                    'message': 'Invalid API token. Please check your token.'
                }
            elif "timeout" in error_message.lower():
                return {
                    'valid': False,
                    'message': 'Connection timeout. Please try again.'
                }
            else:
                return {
                    'valid': False,
                    'message': f'Validation failed: Connection error'
                }
    
    def get_account_info(self, api_token):
        """Get account information from Deriv"""
        try:
            ws = websocket.create_connection(
                self.ws_url,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                timeout=10
            )
            
            # Authorize first
            auth_request = {
                "authorize": api_token,
                "req_id": 1
            }
            ws.send(json.dumps(auth_request))
            ws.settimeout(10)
            auth_response = json.loads(ws.recv())
            
            if 'error' in auth_response:
                ws.close()
                return {
                    'success': False,
                    'message': auth_response['error']['message']
                }
            
            # Get balance
            balance_request = {
                "balance": 1,
                "req_id": 2
            }
            ws.send(json.dumps(balance_request))
            balance_response = json.loads(ws.recv())
            
            ws.close()
            
            if 'error' in balance_response:
                return {
                    'success': False,
                    'message': balance_response['error']['message']
                }
            
            return {
                'success': True,
                'data': {
                    'loginid': auth_response['authorize']['loginid'],
                    'balance': float(balance_response['balance']['balance']),
                    'currency': balance_response['balance']['currency'],
                    'account_type': 'demo' if 'VRTC' in auth_response['authorize']['loginid'] else 'real'
                }
            }
            
        except Exception as e:
            print(f"Get account info error: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to get account info: Connection error'
            }
    
    def get_real_time_balance(self, api_token):
        """Get real-time balance from Deriv"""
        try:
            ws = websocket.create_connection(
                self.ws_url,
                sslopt={"cert_reqs": ssl.CERT_NONE},
                timeout=10
            )
            
            # Authorize
            auth_request = {
                "authorize": api_token,
                "req_id": 1
            }
            ws.send(json.dumps(auth_request))
            ws.settimeout(10)
            auth_response = json.loads(ws.recv())
            
            if 'error' in auth_response:
                ws.close()
                return {
                    'success': False,
                    'message': auth_response['error']['message']
                }
            
            # Get current balance
            balance_request = {
                "balance": 1,
                "req_id": 2
            }
            ws.send(json.dumps(balance_request))
            balance_response = json.loads(ws.recv())
            
            ws.close()
            
            if 'error' in balance_response:
                return {
                    'success': False,
                    'message': balance_response['error']['message']
                }
            
            return {
                'success': True,
                'data': {
                    'balance': float(balance_response['balance']['balance']),
                    'currency': balance_response['balance']['currency'],
                    'loginid': balance_response['balance']['loginid']
                }
            }
            
        except Exception as e:
            print(f"Get real-time balance error: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to get balance: Connection error'
            }
    
    def start_balance_stream(self, api_token, callback):
        """Start real-time balance streaming"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'balance' in data:
                    callback({
                        'balance': float(data['balance']['balance']),
                        'currency': data['balance']['currency'],
                        'loginid': data['balance']['loginid']
                    })
            except Exception as e:
                print(f"Balance stream message error: {str(e)}")
        
        def on_error(ws, error):
            print(f"Balance stream error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            print("Balance stream closed")
        
        def on_open(ws):
            # Authorize and subscribe to balance
            auth_request = {
                "authorize": api_token,
                "req_id": 1
            }
            ws.send(json.dumps(auth_request))
            
            # Subscribe to balance updates
            balance_request = {
                "balance": 1,
                "subscribe": 1,
                "req_id": 2
            }
            ws.send(json.dumps(balance_request))
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
