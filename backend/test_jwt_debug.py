#!/usr/bin/env python3
import requests
import json

# Test login first
login_data = {
    "mobile_number": "+94700000001",  # Changed to be consistent
    "password": "TestPassword123!"
}

print("Step 1: Testing login...")
login_response = requests.post("http://127.0.0.1:5000/api/auth/login", json=login_data)
print(f"Login Status: {login_response.status_code}")

if login_response.status_code == 200:
    login_result = login_response.json()
    print(f"Login response: {json.dumps(login_result, indent=2)}")
    access_token = login_result.get('access_token')
    print(f"Login successful!")
    if access_token:
        print(f"Token (first 50 chars): {access_token[:50]}...")
    else:
        print("WARNING: No access_token in response!")
    
    # Test protected endpoint - profile
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    print("\nStep 2: Testing profile endpoint (to verify JWT works)...")
    profile_response = requests.get("http://127.0.0.1:5000/api/auth/profile", headers=headers)
    print(f"Profile Status: {profile_response.status_code}")
    if profile_response.status_code == 200:
        print("Profile endpoint works - JWT is valid")
    else:
        print(f"Profile failed: {profile_response.text}")
    
    # Test trading test endpoint
    print("\nStep 3: Testing trading test endpoint...")
    trading_test_response = requests.get("http://127.0.0.1:5000/api/trading/test", headers=headers)
    print(f"Trading Test Status: {trading_test_response.status_code}")
    if trading_test_response.status_code == 200:
        print("Trading test endpoint works")
    else:
        print(f"Trading test failed: {trading_test_response.text}")
    
    # Now test the setup-api endpoint
    print("\nStep 4: Testing setup-api endpoint...")
    api_setup_data = {
        "api_token": "test_deriv_api_token_12345",
        "account_type": "demo"
    }
    
    setup_response = requests.post("http://127.0.0.1:5000/api/trading/setup-api", 
                                 json=api_setup_data, headers=headers)
    print(f"Setup API Status: {setup_response.status_code}")
    try:
        response_json = setup_response.json()
        print(f"Response: {json.dumps(response_json, indent=2)}")
    except:
        print(f"Response Text: {setup_response.text}")

else:
    print(f"Login failed: {login_response.text}")
