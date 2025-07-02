#!/usr/bin/env python3
import os
from dotenv import load_dotenv

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

# Try to load .env explicitly
if os.path.exists('.env'):
    print("✓ .env file found in current directory")
    load_dotenv('.env')
else:
    print("✗ .env file not found in current directory")

if os.path.exists('backend/.env'):
    print("✓ .env file found in backend directory")
    load_dotenv('backend/.env')
else:
    print("✗ .env file not found in backend directory")

if os.path.exists('D:/GITHUB/trd_bot/backend/.env'):
    print("✓ .env file found in full path")
    load_dotenv('D:/GITHUB/trd_bot/backend/.env')
else:
    print("✗ .env file not found in full path")

# Check environment variables after loading
print(f"\nEnvironment variables after loading:")
print(f"JWT_SECRET_KEY: {os.getenv('JWT_SECRET_KEY', 'NOT_SET')}")
print(f"SECRET_KEY: {os.getenv('SECRET_KEY', 'NOT_SET')}")
print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'NOT_SET')}")
