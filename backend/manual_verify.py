#!/usr/bin/env python3
"""
Manual verification for testing - this directly marks a user as verified in the database
"""
import sqlite3
from datetime import datetime

# Connect to the database
db_path = "trading_bot.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find the test user
cursor.execute("SELECT id, mobile_number, is_verified FROM user WHERE mobile_number = ?", ("+94700000001",))
user = cursor.fetchone()

if user:
    user_id, mobile, is_verified = user
    print(f"Found user: ID={user_id}, Mobile={mobile}, Verified={is_verified}")
    
    if not is_verified:
        # Mark user as verified
        cursor.execute("""
            UPDATE user 
            SET is_verified = 1, verified_at = ? 
            WHERE id = ?
        """, (datetime.utcnow().isoformat(), user_id))
        
        conn.commit()
        print("User marked as verified!")
    else:
        print("User is already verified!")
else:
    print("User not found!")

conn.close()
