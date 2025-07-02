#!/usr/bin/env python3
import requests
import json

# Test data for registration
register_data = {
    "first_name": "Test",
    "last_name": "User",
    "mobile_number": "+94700000001",  # Changed to be consistent
    "password": "TestPassword123!"
}

print("Step 1: Registering a new user...")
register_response = requests.post("http://127.0.0.1:5000/api/auth/register", json=register_data)
print(f"Register Status Code: {register_response.status_code}")

if register_response.status_code == 201:
    register_result = register_response.json()
    user_id = register_result['user']['id']
    print(f"Registration successful! User ID: {user_id}")
    
    # Step 2: Verify the user (simulate verification)
    verify_data = {
        "user_id": user_id,
        "verification_code": "123456"  # Use the generated code or mock it
    }
    
    print("\nStep 2: Verifying user...")
    verify_response = requests.post("http://127.0.0.1:5000/api/auth/verify", json=verify_data)
    print(f"Verify Status Code: {verify_response.status_code}")
    
    if verify_response.status_code == 200:
        verify_result = verify_response.json()
        access_token = verify_result.get('access_token')
        print(f"Verification successful! Token: {access_token[:20]}...")
        
        # Step 3: Setup API token
        api_setup_data = {
            "api_token": "test_deriv_api_token_12345",
            "account_type": "demo"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        print("\nStep 3: Testing API token setup...")
        response = requests.post("http://127.0.0.1:5000/api/trading/setup-api", 
                               json=api_setup_data, headers=headers)
        
        print(f"Setup API Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
    else:
        print(f"Verification failed: {verify_response.text}")
else:
    print(f"Registration failed: {register_response.text}")
    
    # Try to login with existing user
    print("\nTrying to login with existing user...")
    login_data = {
        "mobile_number": "+94700000001",  # Changed to be consistent
        "password": "TestPassword123!"
    }
    
    login_response = requests.post("http://127.0.0.1:5000/api/auth/login", json=login_data)
    print(f"Login Status Code: {login_response.status_code}")
    
    if login_response.status_code == 200:
        login_result = login_response.json()
        access_token = login_result.get('access_token')
        print(f"Login successful! Token: {access_token[:20]}...")
        
        # Setup API token
        api_setup_data = {
            "api_token": "test_deriv_api_token_12345",
            "account_type": "demo"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        print("\nTesting API token setup...")
        response = requests.post("http://127.0.0.1:5000/api/trading/setup-api", 
                               json=api_setup_data, headers=headers)
        
        print(f"Setup API Response Status Code: {response.status_code}")
        try:
            response_json = response.json()
            print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response Text: {response.text}")
    else:
        print(f"Login failed: {login_response.text}")
