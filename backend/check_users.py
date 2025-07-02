#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models.user import User

app = create_app()

with app.app_context():
    print("Checking database users:")
    users = User.query.all()
    print(f"Total users in database: {len(users)}")
    
    for user in users:
        print(f"ID: {user.id}, Mobile: {user.mobile_number}, Email: {user.email}, Verified: {user.is_verified}")
        print(f"Has API Token: {bool(user.deriv_api_token)}")
    
    # Create test user if none exists
    if len(users) == 0:
        print("\nCreating test user...")
        test_user = User(
            mobile_number="+94712494975",
            first_name="Test",
            last_name="User",
            is_verified=True,
            is_active=True
        )
        test_user.set_password("TestPassword123!")
        
        from models import db
        db.session.add(test_user)
        db.session.commit()
        print("Test user created successfully!")
    else:
        # Update existing user to be verified and active
        for user in users:
            if user.mobile_number == "+94712494975":
                user.is_verified = True
                user.is_active = True
                # Make sure password is set correctly
                user.set_password("TestPassword123!")
                from models import db
                db.session.commit()
                print(f"Updated user {user.id} to be verified and active")
                break
