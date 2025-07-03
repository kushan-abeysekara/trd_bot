#!/usr/bin/env python3
import requests
import json
from datetime import datetime

# Test just the login endpoint
login_data = {
    "mobile_number": "+94700000001",  # Changed to be consistent
    "password": "TestPassword123!"
}

print("Testing login endpoint...")
response = requests.post("http://127.0.0.1:5000/api/auth/login", json=login_data)

print(f"Response Status: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}")

if response.status_code == 200:
    result = response.json()
    access_token = result.get('access_token')
    print(f"Login successful!")
    print(f"Token received: {access_token[:50]}...")
    
    # Test the token immediately with a simple endpoint
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    print("\nTesting token validity with balance endpoint...")
    balance_response = requests.get("http://127.0.0.1:5000/api/trading/balance", headers=headers)
    print(f"Balance endpoint status: {balance_response.status_code}")
    
    try:
        balance_json = balance_response.json()
        print(f"Balance response: {json.dumps(balance_json, indent=2)}")
        
        # Test both account types
        print("\nTesting demo balance...")
        demo_response = requests.get("http://127.0.0.1:5000/api/deriv/balance?account_type=demo", headers=headers)
        print(f"Demo balance status: {demo_response.status_code}")
        try:
            print(f"Demo balance: {json.dumps(demo_response.json(), indent=2)}")
        except:
            print(f"Demo balance text: {demo_response.text}")
        
        print("\nTesting real balance...")
        real_response = requests.get("http://127.0.0.1:5000/api/deriv/balance?account_type=real", headers=headers)
        print(f"Real balance status: {real_response.status_code}")
        try:
            print(f"Real balance: {json.dumps(real_response.json(), indent=2)}")
        except:
            print(f"Real balance text: {real_response.text}")
            
    except:
        print(f"Balance response text: {balance_response.text}")
    
    # Try a different endpoint - health check if it exists
    print("\nTesting health endpoint...")
    health_response = requests.get("http://127.0.0.1:5000/api/health")
    print(f"Health endpoint status: {health_response.status_code}")
    try:
        health_json = health_response.json()
        print(f"Health response: {json.dumps(health_json, indent=2)}")
    except:
        print(f"Health response text: {health_response.text}")
    
else:
    try:
        error_json = response.json()
        print(f"Login failed: {json.dumps(error_json, indent=2)}")
    except:
        print(f"Login failed: {response.text}")
