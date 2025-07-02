#!/usr/bin/env python3
import requests
import json

# First login to get a token
login_data = {
    "mobile_number": "+94700000001",  # Changed to be consistent
    "password": "TestPassword123!"
}

print("Step 1: Logging in...")
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
    
    print("\nStep 2: Testing API token setup...")
    print(f"URL: http://127.0.0.1:5000/api/trading/setup-api")
    print(f"Data: {api_setup_data}")
    
    try:
        response = requests.post("http://127.0.0.1:5000/api/trading/setup-api", 
                               json=api_setup_data, headers=headers)
        
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
        
else:
    print(f"Login failed: {login_response.status_code}")
    print(f"Response: {login_response.text}")
