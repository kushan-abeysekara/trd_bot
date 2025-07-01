from flask import Blueprint

# Create auth blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Import route handlers
from . import auth

__all__ = ['auth_bp']
