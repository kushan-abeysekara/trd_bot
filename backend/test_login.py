#!/usr/bin/env python3
import requests
import json

# Test data - using the same mobile number as in other test files
test_data = {
    "mobile_number": "+94700000001",  # Changed to match other test files
    "password": "TestPassword123!"
}

# Test login endpoint
url = "http://127.0.0.1:5000/api/auth/login"
headers = {
    "Content-Type": "application/json"
}

print("Testing login endpoint...")
print(f"URL: {url}")
print(f"Data: {test_data}")

try:
    # First, let's check if the user exists by trying to register (should fail if exists)
    print("\nStep 1: Checking if user exists...")
    register_data = {
        "first_name": "Test",
        "last_name": "User",
        "mobile_number": "+94700000001",
        "password": "TestPassword123!"
    }
    
    register_response = requests.post("http://127.0.0.1:5000/api/auth/register", json=register_data)
    print(f"Register attempt status: {register_response.status_code}")
    
    if register_response.status_code == 201:
        print("User was just created - need to verify first")
        register_result = register_response.json()
        user_id = register_result['user']['id']
        
        # Mock verification
        verify_data = {
            "user_id": user_id,
            "verification_code": "123456"  # This might fail, but let's try
        }
        verify_response = requests.post("http://127.0.0.1:5000/api/auth/verify", json=verify_data)
        print(f"Verification attempt status: {verify_response.status_code}")
    elif register_response.status_code == 409:
        print("User already exists - proceeding with login test")
    else:
        print(f"Unexpected register response: {register_response.text}")
    
    print("\nStep 2: Testing login...")
    response = requests.post(url, json=test_data, headers=headers)
    
    print(f"\nResponse Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    
    try:
        response_json = response.json()
        print(f"Response JSON: {json.dumps(response_json, indent=2)}")
        
        # If login successful, test the token
        if response.status_code == 200 and 'access_token' in response_json:
            token = response_json['access_token']
            print(f"\nTesting token with profile endpoint...")
            
            auth_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            profile_response = requests.get("http://127.0.0.1:5000/api/auth/profile", headers=auth_headers)
            print(f"Profile endpoint status: {profile_response.status_code}")
            
    except json.JSONDecodeError:
        print(f"Response Text: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
