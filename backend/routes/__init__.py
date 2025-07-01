from flask import Blueprint

# Create auth blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Import route handlers
from . import auth

# Import the auth blueprint directly from the auth module
from .auth import auth_bp

__all__ = ['auth_bp']
