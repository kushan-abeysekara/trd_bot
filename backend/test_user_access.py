#!/usr/bin/env python3
"""Test script to debug ORM vs raw SQLite user access"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models.user import User, db
import sqlite3

def test_user_access():
    """Test both ORM and raw SQLite user access"""
    
    print("Testing User Access Methods...")
    
    # Test 1: Raw SQLite access
    print("\n--- Raw SQLite Access ---")
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, mobile_number, deriv_api_token FROM users LIMIT 5')
        users = cursor.fetchall()
        print(f"Found {len(users)} users via raw SQLite:")
        for user in users:
            token_preview = user[2][:10] + "..." if user[2] else "None"
            print(f"  ID: {user[0]}, Mobile: {user[1]}, Token: {token_preview}")
        conn.close()
    except Exception as e:
        print(f"Raw SQLite error: {str(e)}")
    
    # Test 2: SQLAlchemy ORM access
    print("\n--- SQLAlchemy ORM Access ---")
    try:
        app = create_app()
        with app.app_context():
            users = User.query.limit(5).all()
            print(f"Found {len(users)} users via SQLAlchemy ORM:")
            for user in users:
                token_preview = user.deriv_api_token[:10] + "..." if user.deriv_api_token else "None"
                print(f"  ID: {user.id}, Mobile: {user.mobile_number}, Token: {token_preview}")
                
            # Test specific user ID 9
            print(f"\n--- Testing User ID 9 ---")
            user_9 = User.query.get(9)
            if user_9:
                token_preview = user_9.deriv_api_token[:10] + "..." if user_9.deriv_api_token else "None"
                print(f"User 9 found: Mobile: {user_9.mobile_number}, Token: {token_preview}")
            else:
                print("User 9 not found via ORM")
                
            # Test with filter
            user_9_filter = User.query.filter_by(id=9).first()
            if user_9_filter:
                print("User 9 found via filter_by")
            else:
                print("User 9 not found via filter_by")
                
    except Exception as e:
        print(f"SQLAlchemy ORM error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_user_access()
