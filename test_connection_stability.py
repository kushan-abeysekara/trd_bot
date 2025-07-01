#!/usr/bin/env python3
"""
Test Connection Stability Fix
"""
import asyncio
import logging
import time
from datetime import datetime
from deriv_api import DerivAPI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_connection_stability():
    """Test connection stability over time"""
    logger.info("🧪 Testing Connection Stability Fix")
    logger.info("=" * 50)
    
    # Initialize API
    api = DerivAPI()
    
    # Statistics
    connection_attempts = 0
    successful_connections = 0
    total_uptime = 0
    start_time = datetime.now()
    
    try:
        # Test connection
        logger.info("🔄 Testing initial connection...")
        connection_attempts += 1
        
        success = await api.connect()
        if success:
            successful_connections += 1
            logger.info("✅ Initial connection successful")
            
            # Keep connection alive for 5 minutes to test stability
            logger.info("⏱️ Testing connection stability for 5 minutes...")
            
            test_duration = 300  # 5 minutes
            check_interval = 30   # Check every 30 seconds
            
            for i in range(test_duration // check_interval):
                await asyncio.sleep(check_interval)
                
                # Check if still connected
                if api.is_connected:
                    total_uptime += check_interval
                    logger.info(f"✅ Connection stable ({i+1}/{test_duration//check_interval}) - Uptime: {total_uptime}s")
                    
                    # Test with a simple API call
                    try:
                        result = await api.get_active_symbols()
                        if 'error' not in result:
                            logger.info("✅ API call successful")
                        else:
                            logger.warning(f"⚠️ API call returned error: {result['error']}")
                    except Exception as e:
                        logger.warning(f"⚠️ API call failed: {e}")
                        
                else:
                    logger.error(f"❌ Connection lost at check {i+1}")
                    
                    # Try to reconnect
                    logger.info("🔄 Attempting to reconnect...")
                    connection_attempts += 1
                    
                    success = await api.connect()
                    if success:
                        successful_connections += 1
                        logger.info("✅ Reconnection successful")
                    else:
                        logger.error("❌ Reconnection failed")
                        break
        else:
            logger.error("❌ Initial connection failed")
            
    except KeyboardInterrupt:
        logger.info("🛑 Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
    finally:
        # Disconnect
        await api.disconnect()
        
        # Show statistics
        end_time = datetime.now()
        total_test_time = (end_time - start_time).total_seconds()
        
        logger.info("📊 Connection Stability Test Results")
        logger.info("=" * 50)
        logger.info(f"Total test time: {total_test_time:.1f} seconds")
        logger.info(f"Connection attempts: {connection_attempts}")
        logger.info(f"Successful connections: {successful_connections}")
        logger.info(f"Connection success rate: {(successful_connections/connection_attempts*100):.1f}%")
        logger.info(f"Total uptime: {total_uptime} seconds")
        logger.info(f"Uptime percentage: {(total_uptime/total_test_time*100):.1f}%")
        
        if total_uptime > total_test_time * 0.9:
            logger.info("🎉 EXCELLENT: Connection is very stable!")
        elif total_uptime > total_test_time * 0.7:
            logger.info("✅ GOOD: Connection is reasonably stable")
        elif total_uptime > total_test_time * 0.5:
            logger.info("⚠️ FAIR: Connection has some stability issues")
        else:
            logger.info("❌ POOR: Connection is very unstable")

if __name__ == "__main__":
    print("🔧 Connection Stability Test")
    print("This will test the connection for 5 minutes")
    print("Press Ctrl+C to stop early")
    print("=" * 50)
    
    asyncio.run(test_connection_stability())
