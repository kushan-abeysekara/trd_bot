#!/usr/bin/env python3
import requests
import json

# Login first
login_data = {
    "mobile_number": "+94700000001",
    "password": "TestPassword123!"
}

print("Step 1: Logging in...")
response = requests.post("http://127.0.0.1:5000/api/auth/login", json=login_data)

if response.status_code == 200:
    result = response.json()
    access_token = result.get('access_token')
    print(f"Login successful! Token: {access_token[:50]}...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    print("\nStep 2: Testing debug token endpoint...")
    debug_response = requests.get("http://127.0.0.1:5000/api/trading/debug-token", headers=headers)
    print(f"Debug endpoint status: {debug_response.status_code}")
    
    try:
        debug_json = debug_response.json()
        print(f"Debug response: {json.dumps(debug_json, indent=2)}")
    except:
        print(f"Debug response text: {debug_response.text}")
        
else:
    print(f"Login failed: {response.status_code}")
    try:
        error_json = response.json()
        print(f"Login error: {json.dumps(error_json, indent=2)}")
    except:
        print(f"Login error text: {response.text}")
