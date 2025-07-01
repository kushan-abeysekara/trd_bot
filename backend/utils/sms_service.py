import requests
import os
import random
import string
import json
from datetime import datetime, timedelta
from urllib.parse import quote_plus

class SMSService:
    def __init__(self):
        self.api_url = os.getenv('SMS_API_URL')
        self.api_key = os.getenv('SMS_API_KEY')
        self.api_token = os.getenv('SMS_API_TOKEN')
        self.sender_id = os.getenv('SMS_SENDER_ID')
    
    def generate_verification_code(self):
        """Generate a 6-digit verification code"""
        return ''.join(random.choices(string.digits, k=6))
    
    def send_verification_sms(self, mobile_number, verification_code):
        """Send verification SMS to mobile number"""
        try:
            # Format the message
            message = f"Your TradingBot verification code is: {verification_code}. Valid for 10 minutes. Do not share this code."
            
            # URL encode the message
            encoded_message = quote_plus(message)
            
            # Prepare the request parameters - fix the URL format
            params = {
                'sendsms': '',
                'apikey': self.api_key,
                'apitoken': self.api_token,
                'type': 'sms',
                'from': self.sender_id,
                'to': mobile_number,
                'text': encoded_message,
                'route': '0'
            }
            
            # Build URL with parameters
            url = f"{self.api_url}?sendsms&apikey={self.api_key}&apitoken={self.api_token}&type=sms&from={self.sender_id}&to={mobile_number}&text={encoded_message}&route=0"
            
            print(f"Sending SMS to: {mobile_number}")
            print(f"SMS URL: {url}")
            
            # Send the SMS
            response = requests.get(url, timeout=30)
            
            print(f"SMS Response Status: {response.status_code}")
            print(f"SMS Response Body: {response.text}")
            
            if response.status_code == 200:
                try:
                    # Try to parse as JSON
                    response_data = response.json()
                    
                    # Check if the response indicates success
                    if response_data.get('status') == 'queued':
                        return {
                            'success': True,
                            'message': 'SMS sent successfully and queued for delivery',
                            'response': response_data,
                            'group_id': response_data.get('group_id')
                        }
                    elif response_data.get('status') == 'error':
                        return {
                            'success': False,
                            'message': f"SMS API error: {response_data.get('message', 'Unknown error')}",
                            'error': response_data
                        }
                    else:
                        return {
                            'success': False,
                            'message': f"Unexpected status: {response_data.get('status')}",
                            'error': response_data
                        }
                        
                except json.JSONDecodeError:
                    # If not JSON, check text response
                    response_text = response.text.lower()
                    if 'queued' in response_text or 'success' in response_text or 'sent' in response_text:
                        return {
                            'success': True,
                            'message': 'SMS sent successfully (text response)',
                            'response': response.text
                        }
                    else:
                        return {
                            'success': False,
                            'message': 'SMS sending failed (text response)',
                            'error': response.text
                        }
            else:
                return {
                    'success': False,
                    'message': f'SMS API request failed with status {response.status_code}',
                    'error': response.text
                }
                
        except requests.exceptions.RequestException as e:
            print(f"SMS Request Exception: {str(e)}")
            return {
                'success': False,
                'message': 'SMS service connection error',
                'error': str(e)
            }
        except Exception as e:
            print(f"SMS General Exception: {str(e)}")
            return {
                'success': False,
                'message': 'Unexpected error occurred',
                'error': str(e)
            }
    
    def send_welcome_sms(self, mobile_number, first_name):
        """Send welcome SMS after successful registration"""
        try:
            message = f"Welcome to TradingBot, {first_name}! Your account has been successfully created. Start your trading journey now!"
            encoded_message = quote_plus(message)
            
            url = f"{self.api_url}?sendsms&apikey={self.api_key}&apitoken={self.api_token}&type=sms&from={self.sender_id}&to={mobile_number}&text={encoded_message}&route=0"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return response_data.get('status') == 'queued'
                except json.JSONDecodeError:
                    return 'queued' in response.text.lower() or 'success' in response.text.lower()
            
            return False
            
        except Exception as e:
            print(f"Failed to send welcome SMS: {str(e)}")
            return False
    
    def check_sms_status(self, group_id):
        """Check the status of sent SMS using group_id"""
        try:
            url = f"{self.api_url}?groupstatus&apikey={self.api_key}&apitoken={self.api_token}&groupid={group_id}"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return {
                        'success': True,
                        'status': response_data.get('group_status'),
                        'recipients': response_data.get('recipients', [])
                    }
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'error': 'Invalid JSON response'
                    }
            
            return {
                'success': False,
                'error': f'Request failed with status {response.status_code}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_balance(self):
        """Get SMS credit balance"""
        try:
            url = f"{self.api_url}?balance&apikey={self.api_key}&apitoken={self.api_token}"
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if 'balance' in response_data:
                        return {
                            'success': True,
                            'balance': response_data['balance']
                        }
                    else:
                        return {
                            'success': False,
                            'error': response_data.get('message', 'Unknown error')
                        }
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'error': 'Invalid JSON response'
                    }
            
            return {
                'success': False,
                'error': f'Request failed with status {response.status_code}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
