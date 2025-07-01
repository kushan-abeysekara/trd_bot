#!/usr/bin/env python3
"""
Easy installation and run script for the Deriv AI Trading Bot
"""
import sys
import subprocess
import os
from pathlib import Path

def print_header():
    print("=" * 60)
    print("    🤖 DERIV AI TRADING BOT - EASY SETUP")
    print("=" * 60)
    print()

def check_python():
    """Check Python version"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("❌ Python 3.8+ is required!")
        return False
    
    if version.minor >= 13:
        print("⚠️  Warning: Python 3.13+ may have compatibility issues")
        print("   Recommended: Python 3.8-3.12")
    
    return True

def install_packages():
    """Install packages with fallback options"""
    print("\n📦 Installing packages...")
    
    # Core packages that should install easily
    core_packages = [
        "websockets>=11.0.0",
        "requests>=2.31.0", 
        "python-dotenv>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "sqlalchemy>=1.4.0,<2.0.0",
        "aiofiles>=23.0.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.6",
        "httpx>=0.27.0",
        "pymysql>=1.1.0",
        "schedule>=1.2.0",
        "openai>=1.3.0"
    ]
    
    # Scientific packages (may need special handling)
    scientific_packages = [
        "numpy>=1.21.0,<1.27.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0"
    ]
    
    # Optional packages
    optional_packages = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0", 
        "cryptography>=41.0.0",
        "psutil>=5.9.0"
    ]
    
    # Install core packages
    print("Installing core packages...")
    for package in core_packages:
        try:
            print(f"  Installing {package.split('>=')[0]}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Failed to install {package}")
    
    # Install scientific packages with fallbacks
    print("Installing scientific packages...")
    for package in scientific_packages:
        try:
            print(f"  Installing {package.split('>=')[0]}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--no-build-isolation"], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            # Try without version constraints
            simple_name = package.split('>=')[0]
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", simple_name], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  ✅ Installed {simple_name} (fallback version)")
            except subprocess.CalledProcessError:
                print(f"  ❌ Failed to install {simple_name}")
    
    # Install optional packages (don't fail if these don't work)
    print("Installing optional packages...")
    for package in optional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Optional package {package.split('>=')[0]} failed to install")
    
    print("✅ Package installation completed")

def setup_environment():
    """Setup environment files"""
    print("\n⚙️  Setting up environment...")
    
    # Check if .env exists
    if not Path(".env").exists():
        print("Creating .env file...")
        env_content = """# Environment Configuration
DERIV_TOKEN=your_deriv_token_here
OPENAI_API_KEY=your_openai_key_here

# Database Configuration  
DATABASE_URL=sqlite:///trading_bot.db

# Trading Configuration
DEMO_MODE=true
MAX_DAILY_LOSS=100
MAX_CONSECUTIVE_LOSSES=5  
INITIAL_STAKE=1.0

# WebSocket Configuration
DERIV_WS_URL=wss://ws.binaryws.com/websockets/v3?app_id=1089
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ .env file created (please update with your tokens)")
    else:
        print("✅ .env file already exists")
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    print("✅ Directories created")

def test_imports():
    """Test if critical imports work"""
    print("\n🧪 Testing imports...")
    
    critical_imports = [
        ("websockets", "websockets"),
        ("requests", "requests"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("sqlalchemy", "sqlalchemy"),
        ("numpy", "numpy"),
        ("pandas", "pandas")
    ]
    
    failed_imports = []
    for name, module in critical_imports:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️  Some packages failed to import: {', '.join(failed_imports)}")
        print("The bot may still work with reduced functionality")
    else:
        print("\n✅ All critical packages imported successfully")

def run_bot():
    """Run the trading bot"""
    print("\n" + "=" * 60)
    print("       🚀 STARTING TRADING BOT")
    print("=" * 60)
    
    print("\nWhich component would you like to run?")
    print("1. Trading Bot (main.py)")
    print("2. Web Dashboard (dashboard.py)")
    print("3. Setup and configuration")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🤖 Starting Trading Bot...")
            try:
                subprocess.run([sys.executable, "main.py"])
            except KeyboardInterrupt:
                print("\n👋 Bot stopped by user")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            break
            
        elif choice == "2":
            print("\n🌐 Starting Web Dashboard...")
            print("Dashboard will be available at: http://localhost:8000")
            try:
                subprocess.run([sys.executable, "dashboard.py"])
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped by user")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            break
            
        elif choice == "3":
            print("\n⚙️  Configuration:")
            print("1. Edit .env file with your tokens")
            print("2. Check database connection")
            print("3. Test API connection")
            print("Please manually edit the .env file with your API tokens")
            break
            
        elif choice == "4":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("Please enter 1, 2, 3, or 4")

def main():
    """Main function"""
    print_header()
    
    # Check Python
    if not check_python():
        input("Press Enter to exit...")
        return
    
    # Install packages
    install_packages()
    
    # Setup environment
    setup_environment()
    
    # Test imports
    test_imports()
    
    # Run bot
    run_bot()

if __name__ == "__main__":
    main()
