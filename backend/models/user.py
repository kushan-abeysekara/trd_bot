from flask_sqlalchemy import SQLAlchemy
import bcrypt
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    mobile_number = db.Column(db.String(20), unique=True, nullable=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # Verification fields
    is_verified = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(10), nullable=True)
    verification_code_expires = db.Column(db.DateTime, nullable=True)
    
    # Account status
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Deriv API fields
    deriv_api_token = db.Column(db.Text, nullable=True)  # Keep for backward compatibility
    deriv_demo_api_token = db.Column(db.Text, nullable=True)
    deriv_real_api_token = db.Column(db.Text, nullable=True)
    deriv_account_type = db.Column(db.String(20), default='demo')  # 'demo' or 'real'
    deriv_account_id = db.Column(db.String(50), nullable=True)
    deriv_demo_account_id = db.Column(db.String(50), nullable=True)
    deriv_real_account_id = db.Column(db.String(50), nullable=True)
    deriv_balance = db.Column(db.Float, default=0.0)
    deriv_currency = db.Column(db.String(10), default='USD')
    api_token_created_at = db.Column(db.DateTime, nullable=True)
    api_token_updated_at = db.Column(db.DateTime, nullable=True)
    
    # Trading fields
    trading_balance = db.Column(db.Float, default=0.0)
    auto_trade_enabled = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        """Hash and set password using bcrypt"""
        # Convert password to bytes and hash it
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        self.password_hash = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
    
    def check_password(self, password):
        """Check if password matches hash using bcrypt or fallback to Werkzeug"""
        try:
            # First try bcrypt (new format)
            if self.password_hash.startswith('$2b$'):
                password_bytes = password.encode('utf-8')
                stored_hash_bytes = self.password_hash.encode('utf-8')
                return bcrypt.checkpw(password_bytes, stored_hash_bytes)
            else:
                # Fallback to Werkzeug for old hashes
                from werkzeug.security import check_password_hash
                is_valid = check_password_hash(self.password_hash, password)
                
                # If the old password is valid, migrate to bcrypt
                if is_valid:
                    print(f"Migrating password hash for user {self.id} to bcrypt")
                    self.set_password(password)
                    try:
                        db.session.commit()
                        print("Password hash migration successful")
                    except Exception as e:
                        print(f"Password hash migration failed: {e}")
                        db.session.rollback()
                
                return is_valid
        except (ValueError, TypeError, ImportError) as e:
            print(f"Password check error: {e}")
            return False
    
    def to_dict(self):
        """Convert user to dictionary"""
        try:
            has_demo_token = bool(self.deriv_demo_api_token)
            has_real_token = bool(self.deriv_real_api_token)
            has_any_token = has_demo_token or has_real_token or bool(self.deriv_api_token)
            
            return {
                'id': self.id,
                'email': self.email,
                'mobile_number': self.mobile_number,
                'first_name': self.first_name,
                'last_name': self.last_name,
                'is_verified': self.is_verified,
                'is_active': self.is_active,
                'is_admin': getattr(self, 'is_admin', False),
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'has_api_token': has_any_token,
                'api_configured': has_any_token,
                'deriv_api_token': has_any_token,  # For backward compatibility
                'has_demo_token': has_demo_token,
                'has_real_token': has_real_token,
                'deriv_account_type': self.deriv_account_type,
                'deriv_account_id': self.deriv_account_id,
                'deriv_demo_account_id': self.deriv_demo_account_id,
                'deriv_real_account_id': self.deriv_real_account_id,
                'deriv_balance': self.deriv_balance,
                'deriv_currency': self.deriv_currency,
                'trading_balance': float(self.trading_balance) if self.trading_balance else 0.0,
                'auto_trade_enabled': getattr(self, 'auto_trade_enabled', False),
                'api_token_configured': has_any_token
            }
        except Exception as e:
            print(f"Error converting user to dict: {e}")
            # Return basic user info if advanced fields fail
            return {
                'id': self.id,
                'email': self.email,
                'mobile_number': self.mobile_number,
                'first_name': self.first_name,
                'last_name': self.last_name,
                'is_verified': self.is_verified,
                'is_active': self.is_active,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'has_api_token': False,
                'api_configured': False,
                'has_demo_token': False,
                'has_real_token': False
            }
    
    @staticmethod
    def ensure_database_connection():
        """Ensure database connection is working"""
        try:
            from sqlalchemy import text
            # Simple query to test connection
            db.session.execute(text('SELECT 1'))
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False
    
    @staticmethod
    def migrate_schema():
        """Add new columns if they don't exist"""
        try:
            from sqlalchemy import text
            
            # List of new columns to add
            new_columns = [
                ('deriv_api_token', 'TEXT'),
                ('deriv_demo_api_token', 'TEXT'),
                ('deriv_real_api_token', 'TEXT'),
                ('deriv_account_type', 'VARCHAR(20) DEFAULT "demo"'),
                ('deriv_account_id', 'VARCHAR(50)'),
                ('deriv_demo_account_id', 'VARCHAR(50)'),
                ('deriv_real_account_id', 'VARCHAR(50)'),
                ('deriv_balance', 'FLOAT DEFAULT 0.0'),
                ('deriv_currency', 'VARCHAR(10) DEFAULT "USD"'),
                ('api_token_created_at', 'DATETIME'),
                ('api_token_updated_at', 'DATETIME'),
                ('is_admin', 'BOOLEAN DEFAULT 0'),
                ('trading_balance', 'FLOAT DEFAULT 0.0'),
                ('auto_trade_enabled', 'BOOLEAN DEFAULT 0')
            ]
            
            for column_name, column_type in new_columns:
                try:
                    # Try to add the column using SQLAlchemy 2.0 syntax
                    sql_statement = text(f'ALTER TABLE users ADD COLUMN {column_name} {column_type}')
                    db.session.execute(sql_statement)
                    db.session.commit()
                    print(f"Added column: {column_name}")
                except Exception as column_error:
                    # Column might already exist, rollback and continue
                    db.session.rollback()
                    error_msg = str(column_error).lower()
                    if 'duplicate column' in error_msg or 'already exists' in error_msg:
                        print(f"Column {column_name} already exists")
                    else:
                        print(f"Error adding column {column_name}: {column_error}")
            
            print("Schema migration completed")
            
        except Exception as e:
            print(f"Schema migration error: {str(e)}")
            db.session.rollback()

class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String(255), nullable=False, unique=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))
