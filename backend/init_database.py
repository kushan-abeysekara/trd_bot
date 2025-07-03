#!/usr/bin/env python3
"""Initialize the database with all required tables"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from models import db, User

def init_database():
    """Initialize database with all tables"""
    app = create_app()
    
    with app.app_context():
        try:
            print("Creating database tables...")
            
            # Drop all tables and recreate them
            db.drop_all()
            print("Dropped existing tables")
            
            # Create all tables
            db.create_all()
            print("Created all tables")
            
            # Run any additional schema migrations
            try:
                User.migrate_schema()
                print("Schema migrations completed")
            except Exception as e:
                print(f"Migration error (continuing): {e}")
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"Created tables: {tables}")
            
            # Create a test user if none exists
            if User.query.count() == 0:
                print("Creating test user...")
                test_user = User(
                    first_name="Test",
                    last_name="User",
                    mobile_number="+94712494975",
                    email="test@example.com",
                    is_verified=True,
                    is_active=True,
                    deriv_api_token="test_api_token_12345"  # You'll need to replace this with a real token
                )
                test_user.set_password("password123")
                db.session.add(test_user)
                db.session.commit()
                print("Test user created successfully")
            
            # Final verification with SQLite
            import sqlite3
            conn = sqlite3.connect('trading_bot.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            sqlite_tables = [row[0] for row in cursor.fetchall()]
            print(f"Final SQLite tables: {sqlite_tables}")
            
            # Check if users table exists and has data
            try:
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                print(f"Users in database: {user_count}")
            except Exception as e:
                print(f"Error checking users table: {e}")
            
            conn.close()
            
            print("✓ Database initialization completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Database initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = init_database()
    if success:
        print("\n✓ Database is ready for the trading bot!")
    else:
        print("\n✗ Database initialization failed!")
        sys.exit(1)
