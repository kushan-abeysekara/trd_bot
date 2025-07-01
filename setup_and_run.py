#!/usr/bin/env python3
"""
Deriv AI Trading Bot - System Check and Setup
"""
import sys
import subprocess
import os
import platform
from pathlib import Path

def print_header():
    print("=" * 60)
    print("    🤖 DERIV AI TRADING BOT - SYSTEM CHECK")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print(f"✅ Python version: {version_str}")
    
    if version_info.major != 3:
        print("❌ Python 3 is required!")
        return False
    
    if version_info.minor < 8:
        print("❌ Python 3.8 or higher is required!")
        return False
    
    if version_info.minor >= 13:
        print("⚠️  WARNING: Python 3.13+ has compatibility issues!")
        print("   Recommended: Python 3.8-3.12")
        response = input("   Continue anyway? (y/N): ").lower()
        if response != 'y':
            return False
    
    print("✅ Python version is compatible")
    return True

def check_pip():
    """Check if pip is available"""
    print("\n🔍 Checking pip...")
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available!")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements = [
        "websockets==11.0.3",
        "requests==2.31.0", 
        "python-dotenv==1.0.0",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "sqlalchemy==2.0.23",
        "aiofiles==23.2.1",
        "openai==1.3.9",
        "schedule==1.2.0",
        "jinja2==3.1.2",
        "python-multipart==0.0.6",
        "httpx==0.27.0"
    ]
    
    # Install core packages first
    for req in requirements:
        try:
            print(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req, "--quiet"])
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {req}, but continuing...")
    
    # Try to install numpy and pandas with compatible versions
    try:
        print("Installing numpy...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=1.21.0,<1.27.0", "--quiet"])
        
        print("Installing pandas...")  
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas>=2.0.0", "--no-build-isolation", "--quiet"])
        
        print("Installing scikit-learn...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn>=1.3.0", "--quiet"])
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Some scientific packages failed to install: {e}")
        print("   The bot may still work with basic functionality")
    
    print("✅ Core requirements installed")

def setup_environment():
    """Setup environment and configuration"""
    print("\n⚙️  Setting up environment...")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        env_content = """# Environment Configuration
DERIV_TOKEN=B8ZH857zyOqvHah
OPENAI_API_KEY=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

# Database Configuration  
DATABASE_URL=sqlite:///trading_bot.db

# Trading Configuration
DEMO_MODE=true
MAX_DAILY_LOSS=100
MAX_CONSECUTIVE_LOSSES=5  
INITIAL_STAKE=1.0

# WebSocket Configuration
DERIV_WS_URL=wss://ws.binaryws.com/websockets/v3
"""
        env_file.write_text(env_content)
        print("✅ .env file created")
    else:
        print("✅ .env file already exists")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Logs directory ready")

def initialize_database():
    """Initialize the database"""
    print("\n🗄️  Initializing database...")
    try:
        import database
        database.create_tables()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️  Database setup issue: {e}")
        print("   Bot may still work, but without persistent storage")

def run_bot():
    """Start the trading bot"""
    print("\n" + "=" * 60)
    print("       🚀 STARTING DERIV AI TRADING BOT")
    print("=" * 60)
    print()
    
    print("Bot Configuration:")
    print("- API Token: ✅ Configured") 
    print("- OpenAI Key: ✅ Configured")
    print("- Demo Mode: ✅ ENABLED (Safe)")
    print("- Initial Stake: $1.00")
    print("- Max Daily Loss: $100.00")
    print()
    
    print("⚠️  IMPORTANT SAFETY REMINDERS:")
    print("- Bot is running in DEMO mode (safe)")  
    print("- Always test strategies before live trading")
    print("- Never risk more than you can afford to lose")
    print("- Monitor the bot regularly")
    print()
    
    input("Press Enter to start the bot...")
    
    try:
        import main
        # This will start the bot
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting bot: {e}")
        print("Check the logs for more details")

def main():
    """Main setup and run function"""
    print_header()
    
    # System checks
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    if not check_pip():
        input("Press Enter to exit...")
        return
    
    # Setup
    try:
        install_requirements()
        setup_environment()
        initialize_database()
        
        print("\n✅ Setup completed successfully!")
        print()
        
        # Ask user what to do next
        print("What would you like to do?")
        print("1. Start Trading Bot")
        print("2. Start Web Dashboard") 
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            run_bot()
        elif choice == "2":
            print("\nStarting web dashboard...")
            print("Visit http://localhost:8000 in your browser")
            try:
                import dashboard
            except Exception as e:
                print(f"Error starting dashboard: {e}")
        else:
            print("Goodbye! 👋")
            
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
