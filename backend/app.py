from flask import Flask, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import models and routes
from models import db
from routes.auth import auth_bp
from routes.trading import trading_bp
from routes.deriv_api import deriv_bp

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///trading_bot.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-string')
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 86400))  # 24 hours
    
    # Initialize extensions
    db.init_app(app)
    jwt = JWTManager(app)
    
    # Configure CORS
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    CORS(app, origins=cors_origins, supports_credentials=True, 
         allow_headers=['Content-Type', 'Authorization'],
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
    
    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(trading_bp)
    app.register_blueprint(deriv_bp)
    
    # Create database tables and handle migrations
    with app.app_context():
        try:
            # Import User here to avoid circular imports
            from models.user import User
            
            # Create tables first
            db.create_all()
            print("Database tables created successfully")
            
            # Then run schema migrations
            try:
                User.migrate_schema()
                print("Schema migrations completed successfully")
            except Exception as migration_error:
                print(f"Migration error (continuing anyway): {migration_error}")
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            print("Continuing with application startup...")
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        print(f"Internal server error: {str(error)}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return jsonify({'error': 'Token has expired'}), 401
    
    @jwt.invalid_token_loader
    def invalid_token_callback(error):
        return jsonify({'error': 'Invalid token'}), 401
    
    @jwt.unauthorized_loader
    def missing_token_callback(error):
        return jsonify({'error': 'Authorization token is required'}), 401
    
    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        health_status = {
            'status': 'healthy',
            'message': 'TradingBot API is running',
            'endpoints': {
                'auth': '/api/auth',
                'register': '/api/auth/register',
                'login': '/api/auth/login',
                'verify': '/api/auth/verify',
                'trading': '/api/trading',
                'setup_api': '/api/trading/setup-api',
                'balance': '/api/trading/balance'
            }
        }
        
        # Test database connection
        try:
            from models.user import User
            if User.ensure_database_connection():
                health_status['database'] = 'connected'
            else:
                health_status['database'] = 'disconnected'
                health_status['status'] = 'degraded'
        except Exception as e:
            health_status['database'] = 'error'
            health_status['database_error'] = str(e)
            health_status['status'] = 'degraded'
        
        return jsonify(health_status), 200
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.getenv('PORT', 5000))
    # Disable auto-reload to prevent JWT token invalidation during development
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
