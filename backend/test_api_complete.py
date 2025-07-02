#!/usr/bin/env python3
import requests
import json

# First register a new user
register_data = {
    "first_name": "Test",
    "last_name": "User",
    "mobile_number": "+94700000001",
    "password": "TestPassword123!"
}

print("Step 1: Registering a new user...")
register_response = requests.post("http://127.0.0.1:5000/api/auth/register", json=register_data)
print(f"Register response: {register_response.status_code}")
if register_response.status_code == 201:
    print("Registration successful!")
else:
    print(f"Registration response: {register_response.text}")

# Verify the user (for testing purposes, let's assume verification succeeds)
print("\nStep 2: Logging in...")
login_data = {
    "mobile_number": "+94700000001",
    "password": "TestPassword123!"
}
login_response = requests.post("http://127.0.0.1:5000/api/auth/login", json=login_data)

if login_response.status_code == 200:
    login_result = login_response.json()
    access_token = login_result.get('access_token')
    print(f"Login successful! Token: {access_token[:20]}...")
    
    # Now test the API token setup
    api_setup_data = {
        "api_token": "test_deriv_api_token_12345",
        "account_type": "demo"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    print("\nStep 3: Testing API token setup...")
    print(f"URL: http://127.0.0.1:5000/api/trading/setup-api")
    print(f"Data: {api_setup_data}")
    
    try:
        response = requests.post("http://127.0.0.1:5000/api/trading/setup-api", 
                               json=api_setup_data, headers=headers)
        
        print(f"\nResponse Status Code: {response.status_code}")
        
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
            
        # Test if the token persists by checking balance
        print("\nStep 4: Testing if token persists - checking balance...")
        balance_response = requests.get("http://127.0.0.1:5000/api/trading/balance", headers=headers)
        print(f"Balance check status: {balance_response.status_code}")
        try:
            balance_json = balance_response.json()
            print(f"Balance response: {json.dumps(balance_json, indent=2)}")
        except:
            print(f"Balance response text: {balance_response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
        
else:
    print(f"Login failed: {login_response.status_code}")
    print(f"Response: {login_response.text}")
