#!/usr/bin/env python3
"""
Check database structure
"""
import sqlite3

# Connect to the database
db_path = "trading_bot.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("Tables in database:")
for table in tables:
    print(f"  - {table[0]}")

# Check the user table structure if it exists
if any('user' in str(table).lower() for table in tables):
    # Find the exact table name
    for table in tables:
        table_name = table[0]
        if 'user' in table_name.lower():
            print(f"\nTable '{table_name}' structure:")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Show some sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            print(f"\nSample data from {table_name}:")
            for row in rows:
                print(f"  {row}")

conn.close()
