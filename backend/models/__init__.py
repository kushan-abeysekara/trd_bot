from .user import db, User, UserSession
from datetime import datetime

# Create DerivAccount model here since it needs to reference User
class DerivAccount(db.Model):
    __tablename__ = 'deriv_accounts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    account_id = db.Column(db.String(50), nullable=False)
    account_type = db.Column(db.String(20), nullable=False)  # 'demo' or 'real'
    balance = db.Column(db.Float, default=0.0)
    currency = db.Column(db.String(10), default='USD')
    is_active = db.Column(db.Boolean, default=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('deriv_accounts', lazy=True))
    
    def to_dict(self):
        return {
            'id': self.id,
            'account_id': self.account_id,
            'account_type': self.account_type,
            'balance': self.balance,
            'currency': self.currency,
            'is_active': self.is_active,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

__all__ = ['User', 'UserSession', 'DerivAccount', 'db']
