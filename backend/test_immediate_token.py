#!/usr/bin/env python3
import requests
import json
import jwt
import os
from datetime import datetime

# Login and immediately test token
login_data = {
    "mobile_number": "+94700000001",
    "password": "TestPassword123!"
}

print("Testing immediate token validation...")
session = requests.Session()

# Login
response = session.post("http://127.0.0.1:5000/api/auth/login", json=login_data)

if response.status_code == 200:
    result = response.json()
    access_token = result.get('access_token')
    print(f"✓ Login successful! Token: {access_token[:50]}...")
    
    # Decode the token to see what's inside (for debugging)
    try:
        # Try to decode without verification first to see the payload
        decoded_payload = jwt.decode(access_token, options={"verify_signature": False})
        print(f"✓ Token payload: {json.dumps(decoded_payload, indent=2, default=str)}")
        
        # Check token expiration
        exp = decoded_payload.get('exp')
        if exp:
            exp_time = datetime.fromtimestamp(exp)
            print(f"✓ Token expires at: {exp_time}")
            print(f"✓ Current time: {datetime.now()}")
            print(f"✓ Token valid for: {exp_time - datetime.now()}")
    except Exception as decode_error:
        print(f"✗ Token decode error: {decode_error}")
    
    # Test token immediately with the same session
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    print("\n1. Testing with same session...")
    debug_response = session.get("http://127.0.0.1:5000/api/trading/debug-token", headers=headers)
    print(f"Same session status: {debug_response.status_code}")
    
    print("\n2. Testing with new session...")
    new_response = requests.get("http://127.0.0.1:5000/api/trading/debug-token", headers=headers)
    print(f"New session status: {new_response.status_code}")
    
    # Show response details
    try:
        debug_json = new_response.json()
        print(f"Response: {json.dumps(debug_json, indent=2)}")
    except:
        print(f"Response text: {new_response.text}")
        
else:
    print(f"✗ Login failed: {response.status_code}")
    print(f"Response: {response.text}")
