#!/usr/bin/env python3
"""Test script to debug User model and database connection"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing database connection...")
    
    # Test raw SQLite connection first
    import sqlite3
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    print(f"✓ Raw SQLite connection works. Total users: {user_count}")
    
    # Get a user with API token
    cursor.execute("SELECT id, deriv_api_token FROM users WHERE deriv_api_token IS NOT NULL LIMIT 1")
    user_data = cursor.fetchone()
    if user_data:
        print(f"✓ Found user with API token: ID {user_data[0]}")
    else:
        print("✗ No users with API tokens found")
    
    conn.close()
    
    # Test Flask app context
    from app import app
    with app.app_context():
        print("✓ Flask app context created")
        
        # Test User model import
        from models import User, db
        print("✓ User model imported successfully")
        
        # Test database query
        user_count_orm = User.query.count()
        print(f"✓ SQLAlchemy ORM works. Total users: {user_count_orm}")
        
        # Get user with API token
        user_with_token = User.query.filter(User.deriv_api_token.isnot(None)).first()
        if user_with_token:
            print(f"✓ Found user with API token via ORM: ID {user_with_token.id}")
            print(f"  API token: {user_with_token.deriv_api_token[:20] if user_with_token.deriv_api_token else 'None'}...")
        else:
            print("✗ No users with API tokens found via ORM")

except Exception as e:
    print(f"✗ Error: {str(e)}")
    import traceback
    traceback.print_exc()
