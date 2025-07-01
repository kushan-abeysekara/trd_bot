"""
Database models for the trading bot
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import config
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    contract_id = Column(String(255), unique=True)
    symbol = Column(String(50), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'CALL' or 'PUT'
    stake = Column(Float, nullable=False)
    entry_price = Column(Float)
    exit_price = Column(Float)
    profit_loss = Column(Float)
    duration = Column(Integer)  # in seconds
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String(20), default='PENDING')  # PENDING, WON, LOST
    ai_confidence = Column(Float)  # AI prediction confidence
    technical_signals = Column(Text)  # JSON string of technical indicators
    martingale_level = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class TradingSession(Base):
    __tablename__ = 'trading_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_profit_loss = Column(Float, default=0.0)
    max_consecutive_losses = Column(Integer, default=0)
    current_consecutive_losses = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class AIAnalysis(Base):
    __tablename__ = 'ai_analysis'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    timeframe = Column(String(10), nullable=False)
    analysis_text = Column(Text)
    prediction = Column(String(10))  # 'CALL' or 'PUT'
    confidence = Column(Float)
    technical_data = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)

class BotSettings(Base):
    __tablename__ = 'bot_settings'
    
    id = Column(Integer, primary_key=True)
    setting_name = Column(String(100), unique=True)
    setting_value = Column(String(500))
    updated_at = Column(DateTime, default=datetime.utcnow)

# Database setup
try:
    # Enhanced engine configuration for better MySQL compatibility
    if 'mysql' in config.DATABASE_URL:
        # MySQL-specific configuration
        engine = create_engine(
            config.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
            connect_args={
                'charset': 'utf8mb4',
                'autocommit': True
            }
        )
        logger.info("✅ MySQL database engine configured")
    else:
        # SQLite configuration (fallback)
        engine = create_engine(config.DATABASE_URL, echo=False)
        logger.info("✅ SQLite database engine configured")
        
except Exception as e:
    logger.error(f"❌ Database engine configuration failed: {e}")
    # Fallback to SQLite
    engine = create_engine('sqlite:///trading_bot_fallback.db', echo=False)
    logger.warning("⚠️  Using fallback SQLite database")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully!")
        
        # Test the connection
        test_db = SessionLocal()
        try:
            # Try a simple query to verify connection
            test_db.execute("SELECT 1")
            logger.info("✅ Database connection test successful")
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            raise
        finally:
            test_db.close()
            
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    try:
        create_tables()
        print("🎉 Database setup completed successfully!")
        print(f"📊 Using database: {config.DATABASE_URL}")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("Please check your database configuration and connection.")
