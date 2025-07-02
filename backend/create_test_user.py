#!/usr/bin/env python3
"""
Create and verify a test user using Flask app context
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models import db, User
from datetime import datetime

app = create_app()

with app.app_context():
    # Check if user exists
    existing_user = User.query.filter_by(mobile_number="+94700000001").first()
    
    if existing_user:
        print(f"User exists: ID={existing_user.id}, Verified={existing_user.is_verified}")
        
        if not existing_user.is_verified:
            # Verify the user
            existing_user.is_verified = True
            existing_user.verified_at = datetime.utcnow()
            db.session.commit()
            print("User verified successfully!")
        else:
            print("User is already verified!")
    else:
        print("User not found, creating new user...")
        
        # Create a new verified user for testing
        user = User(
            first_name="Test",
            last_name="User",
            mobile_number="+94700000001",
            password="TestPassword123!",
            is_verified=True,
            verified_at=datetime.utcnow()
        )
        
        db.session.add(user)
        db.session.commit()
        print(f"Test user created and verified! ID={user.id}")
