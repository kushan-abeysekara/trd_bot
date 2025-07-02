import requests
import os
import random
import string
import json
from datetime import datetime, timedelta
from urllib.parse import quote_plus

class SMSService:
    """Service for sending SMS messages"""
    
    def __init__(self):
        # Initialize SMS service (would use Twilio, AWS SNS, etc. in production)
        self.enabled = False  # Set to True when SMS provider is configured
        self.api_url = os.getenv('SMS_API_URL')
        self.api_key = os.getenv('SMS_API_KEY')
        self.api_token = os.getenv('SMS_API_TOKEN')
        self.sender_id = os.getenv('SMS_SENDER_ID')
    
    def generate_verification_code(self, length=6):
        """Generate a random verification code"""
        return ''.join(random.choices(string.digits, k=length))
    
    def send_verification_sms(self, mobile_number, verification_code):
        """Send verification SMS to mobile number"""
        try:
            message = f"Your verification code is: {verification_code}. This code will expire in 10 minutes."
            encoded_message = quote_plus(message)
            
            if self.enabled and self.api_url and self.api_key:
                url = f"{self.api_url}?sendsms&apikey={self.api_key}&apitoken={self.api_token}&type=sms&from={self.sender_id}&to={mobile_number}&text={encoded_message}&route=0"
                
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        return {
                            'success': True,
                            'message': 'SMS sent successfully',
                            'response': response_data
                        }
                    except json.JSONDecodeError:
                        if 'queued' in response.text.lower() or 'success' in response.text.lower():
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
            else:
                # For development, just log the message
                print(f"SMS to {mobile_number}: {message}")
                return {
                    'success': True, 
                    'message': 'SMS sent successfully (development mode)',
                    'verification_code': verification_code  # Only for development
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
            
            if self.enabled and self.api_url and self.api_key:
                url = f"{self.api_url}?sendsms&apikey={self.api_key}&apitoken={self.api_token}&type=sms&from={self.sender_id}&to={mobile_number}&text={encoded_message}&route=0"
                
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        return response_data.get('status') == 'queued'
                    except json.JSONDecodeError:
                        return 'queued' in response.text.lower() or 'success' in response.text.lower()
            else:
                print(f"Welcome SMS to {mobile_number}: {message}")
                return True
            
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
    
    def send_password_reset_sms(self, mobile_number, reset_code):
        """Send password reset SMS"""
        try:
            message = f"Your TradingBot password reset code is: {reset_code}. Valid for 15 minutes."
            encoded_message = quote_plus(message)
            
            if self.enabled and self.api_url and self.api_key:
                url = f"{self.api_url}?sendsms&apikey={self.api_key}&apitoken={self.api_token}&type=sms&from={self.sender_id}&to={mobile_number}&text={encoded_message}&route=0"
                
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        return {
                            'success': True,
                            'message': 'SMS sent successfully',
                            'response': response_data
                        }
                    except json.JSONDecodeError:
                        if 'queued' in response.text.lower() or 'success' in response.text.lower():
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
            else:
                print(f"[SMS SERVICE] Password reset SMS to {mobile_number}: {message}")
                return {
                    'success': True,
                    'message': 'Password reset SMS sent successfully (development mode)'
                }
                
        except requests.exceptions.RequestException as e:
            print(f"SMS Request Exception: {str(e)}")
            return {
                'success': False,
                'message': 'SMS service connection error',
                'error': str(e)
            }
        except Exception as e:
            print(f"Password reset SMS error: {str(e)}")
            return {
                'success': False,
                'message': f'Failed to send password reset SMS: {str(e)}'
            }
