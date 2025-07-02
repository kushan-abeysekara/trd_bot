#!/usr/bin/env python3
import jwt
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test JWT token validation manually
login_data = {
    "mobile_number": "+94700000001",
    "password": "TestPassword123!"
}

print("Testing JWT token validation manually...")

# Login first
response = requests.post("http://127.0.0.1:5000/api/auth/login", json=login_data)

if response.status_code == 200:
    result = response.json()
    access_token = result.get('access_token')
    print(f"✓ Token received: {access_token[:50]}...")
    
    # Try to validate with the actual secret from environment
    jwt_secret = os.getenv('JWT_SECRET_KEY')
    if jwt_secret:
        try:
            decoded = jwt.decode(access_token, jwt_secret, algorithms=['HS256'])
            print(f"✓ Token validated with JWT_SECRET_KEY!")
            print(f"✓ Decoded payload: {json.dumps(decoded, indent=2, default=str)}")
        except jwt.InvalidTokenError as e:
            print(f"✗ Failed with JWT_SECRET_KEY: {e}")
    
    # Try to validate with different secret keys
    potential_secrets = [
        'jwt-secret-string-12345',  # From .env
        'jwt-secret-string',        # Default fallback
        'prod_jwt_2024_trading_bot_super_secret_key_change_this_to_random_64_chars',  # From .env
        'dev-secret-key-12345',     # SECRET_KEY from .env
        'dev-secret-key',           # Default SECRET_KEY
    ]
    
    for secret in potential_secrets:
        try:
            decoded = jwt.decode(access_token, secret, algorithms=['HS256'])
            print(f"✓ Token validated with secret: {secret[:20]}...")
            print(f"✓ Decoded payload: {json.dumps(decoded, indent=2, default=str)}")
            break
        except jwt.InvalidTokenError as e:
            print(f"✗ Failed with secret {secret[:20]}...: {e}")
    
    # Check what Flask-JWT-Extended expects
    print(f"\nChecking environment variables:")
    print(f"JWT_SECRET_KEY from env: {os.getenv('JWT_SECRET_KEY', 'NOT_SET')}")
    print(f"SECRET_KEY from env: {os.getenv('SECRET_KEY', 'NOT_SET')}")
    
else:
    print(f"✗ Login failed: {response.status_code}")
    print(f"Response: {response.text}")
