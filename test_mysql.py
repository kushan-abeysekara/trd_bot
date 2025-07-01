#!/usr/bin/env python3
"""
Test MySQL Database Connection
"""
import os
from dotenv import load_dotenv

# Force load environment variables
load_dotenv(override=True)

print("🔍 Environment Variables Check:")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
print(f"DERIV_TOKEN: {os.getenv('DERIV_TOKEN')[:10]}..." if os.getenv('DERIV_TOKEN') else "DERIV_TOKEN: Not found")

# Test MySQL connection
DATABASE_URL = "mysql+pymysql://u626686198_tradingu:Kushan%4020001018@82.197.82.97:3306/u626686198_trading"

print(f"\n🗄️  Testing MySQL Connection:")
print(f"Host: 82.197.82.97")
print(f"Database: u626686198_trading")
print(f"Username: u626686198_tradingu")

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    
    # Create engine with MySQL connection
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            'charset': 'utf8mb4'
        }
    )
    
    print("✅ MySQL engine created successfully")
    
    # Test connection
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1 as test"))
        test_value = result.fetchone()[0]
        print(f"✅ Database connection successful! Test query returned: {test_value}")
        
        # Check existing tables
        result = connection.execute(text("SHOW TABLES"))
        tables = result.fetchall()
        print(f"📊 Existing tables: {[table[0] for table in tables]}")
    
    # Now try to create our tables
    from database import Base
    print("\n🏗️  Creating trading bot tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created successfully!")
    
    # Verify tables were created
    with engine.connect() as connection:
        result = connection.execute(text("SHOW TABLES"))
        tables = result.fetchall()
        print(f"📋 Current tables: {[table[0] for table in tables]}")
    
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check if the MySQL server is accessible")
    print("2. Verify the credentials are correct")
    print("3. Ensure the database exists")
    print("4. Check firewall settings")
