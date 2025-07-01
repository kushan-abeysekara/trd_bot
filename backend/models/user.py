from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Numeric
from datetime import datetime
import bcrypt

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=True, index=True)
    mobile_number = db.Column(db.String(20), unique=True, nullable=True, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    verification_code = db.Column(db.String(6), nullable=True)
    verification_code_expires = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Trading related fields
    deriv_account_id = db.Column(db.String(100), nullable=True)
    trading_balance = db.Column(Numeric(10, 2), default=0.00)
    auto_trade_enabled = db.Column(db.Boolean, default=False)
    
    def set_password(self, password):
        """Hash and set the password"""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        """Check if provided password matches the hash"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'mobile_number': self.mobile_number,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'is_verified': self.is_verified,
            'is_active': self.is_active,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'deriv_account_id': self.deriv_account_id,
            'trading_balance': float(self.trading_balance) if self.trading_balance else 0.00,
            'auto_trade_enabled': self.auto_trade_enabled
        }

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
