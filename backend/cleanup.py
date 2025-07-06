"""
Cleanup utilities for Deriv Trading Bot
Handles graceful shutdown and resource cleanup
"""
import signal
import sys
import time
import atexit

class GracefulShutdown:
    """Class to handle graceful shutdown of bot and API connections"""
    
    def __init__(self):
        self.bot_instance = None
        self.shutdown_handlers = []
        self.is_shutting_down = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self.handle_shutdown_signal)
        
        # Register with atexit
        atexit.register(self.cleanup)
    
    def register_bot(self, bot):
        """Register bot instance for cleanup"""
        self.bot_instance = bot
    
    def register_handler(self, handler_func):
        """Register additional cleanup handler"""
        if callable(handler_func):
            self.shutdown_handlers.append(handler_func)
    
    def handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals like Ctrl+C"""
        print(f"\n🛑 Received shutdown signal {signum}. Cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Perform cleanup operations"""
        if self.is_shutting_down:
            return
            
        self.is_shutting_down = True
        print("🧹 Performing cleanup operations...")
        
        # Stop bot trading if active
        if self.bot_instance:
            try:
                if hasattr(self.bot_instance, 'is_running') and self.bot_instance.is_running:
                    print("🛑 Stopping trading bot...")
                    self.bot_instance.stop_trading()
                
                # Wait a moment for trading to stop
                time.sleep(0.5)
                
                print("🔌 Disconnecting from API...")
                self.bot_instance.disconnect()
                print("✅ Successfully disconnected from API")
            except Exception as e:
                print(f"❌ Error during bot cleanup: {e}")
        
        # Call additional registered handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                print(f"❌ Error in shutdown handler: {e}")
        
        print("✅ Cleanup complete")

# Create global instance
graceful_shutdown = GracefulShutdown()

def register_bot(bot):
    """Register bot instance with cleanup handler"""
    graceful_shutdown.register_bot(bot)

def register_handler(handler_func):
    """Register cleanup handler function"""
    graceful_shutdown.register_handler(handler_func)
