import requests
import os
import random
import string
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
            
            # Prepare the request parameters
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
            
            # Send the SMS
            response = requests.get(self.api_url, params=params, timeout=30)
            
            if response.status_code == 200:
                response_data = response.text
                # Check if the response indicates success
                if 'success' in response_data.lower() or 'sent' in response_data.lower():
                    return {
                        'success': True,
                        'message': 'SMS sent successfully',
                        'response': response_data
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Failed to send SMS',
                        'error': response_data
                    }
            else:
                return {
                    'success': False,
                    'message': f'SMS API request failed with status {response.status_code}',
                    'error': response.text
                }
                
        except requests.exceptions.RequestException as e:                return {
                    'success': False,
                    'message': 'SMS service connection error',
                    'error': str(e)
                }
        except Exception as e:
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
            
            response = requests.get(self.api_url, params=params, timeout=30)
            return response.status_code == 200
            
        except Exception as e:
            print(f"Failed to send welcome SMS: {str(e)}")
            return False
