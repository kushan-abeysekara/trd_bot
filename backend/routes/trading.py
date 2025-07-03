from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import requests
import json
import random
from datetime import datetime

from models import User, db

trading_bp = Blueprint('trading', __name__, url_prefix='/api/trading')

@trading_bp.route('/setup-api', methods=['POST'])
@jwt_required()
def setup_api_token():
    """Setup Deriv API token"""
    try:
        user_id_str = get_jwt_identity()
        user_id = int(user_id_str)  # Convert string back to int for database query
        print(f"Setting up API token for user ID: {user_id}")
        
        user = User.query.get(user_id)
        
        if not user:
            print(f"User not found with ID: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        print(f"Received data: {data}")
        
        api_token = data.get('api_token')
        account_type = data.get('account_type', 'demo')  # 'demo' or 'real'
        
        print(f"API Token: {api_token[:10]}... (truncated)")
        print(f"Account Type: {account_type}")
        
        if not api_token:
            return jsonify({'error': 'API token is required'}), 400
        
        if account_type not in ['demo', 'real']:
            return jsonify({'error': 'Account type must be demo or real'}), 400
        
        # Validate API token by making a test request
        print("Validating API token...")
        validation_result = validate_deriv_token(api_token)
        print(f"Validation result: {validation_result}")
        
        if not validation_result['success']:
            return jsonify({'error': validation_result['message']}), 400
        
        # Update user with API token
        print("Updating user with API token...")
        user.deriv_api_token = api_token
        user.deriv_account_type = account_type
        user.deriv_account_id = validation_result.get('account_id')
        user.api_token_created_at = datetime.utcnow()
        user.api_token_updated_at = datetime.utcnow()
        
        db.session.commit()
        print("API token saved successfully!")
        
        return jsonify({
            'message': 'API token configured successfully',
            'account_type': account_type,
            'account_id': validation_result.get('account_id')
        }), 200
        
    except Exception as e:
        print(f"Error setting up API token: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to setup API token', 'details': str(e)}), 500

@trading_bp.route('/balance', methods=['GET'])
@jwt_required()
def get_account_balance():
    """Get real-time Deriv account balance"""
    try:
        user_id_str = get_jwt_identity()
        user_id = int(user_id_str)  # Convert string back to int for database query
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if not user.deriv_api_token:
            return jsonify({'error': 'API token not configured'}), 400
        
        # Fetch balance from Deriv API
        balance_result = get_deriv_balance(user.deriv_api_token)
        if not balance_result['success']:
            return jsonify({'error': balance_result['message']}), 400
        
        # Update trading balance in database
        user.trading_balance = balance_result['balance']
        db.session.commit()
        
        return jsonify({
            'balance': balance_result['balance'],
            'currency': balance_result['currency'],
            'account_type': user.deriv_account_type,
            'account_id': user.deriv_account_id,
            'last_updated': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({'error': 'Failed to fetch balance', 'details': str(e)}), 500

@trading_bp.route('/remove-api', methods=['DELETE'])
@jwt_required()
def remove_api_token():
    """Remove Deriv API token"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user.deriv_api_token = None
        user.deriv_account_id = None
        user.deriv_account_type = 'demo'
        user.auto_trade_enabled = False
        db.session.commit()
        
        return jsonify({'message': 'API token removed successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Failed to remove API token', 'details': str(e)}), 500

@trading_bp.route('/test', methods=['GET'])
@jwt_required()
def test_trading():
    """Test trading endpoint"""
    return jsonify({'message': 'Trading routes working'}), 200

@trading_bp.route('/debug-token', methods=['GET'])
@jwt_required()
def debug_token():
    """Debug endpoint to test JWT token validation"""
    try:
        user_id_str = get_jwt_identity()
        user_id = int(user_id_str)  # Convert string back to int for database query
        print(f"DEBUG: Token validation successful, user ID: {user_id}")
        
        user = User.query.get(user_id)
        if user:
            return jsonify({
                'message': 'Token is valid',
                'user_id': user_id,
                'user_mobile': user.mobile_number,
                'user_name': f"{user.first_name} {user.last_name}"
            }), 200
        else:
            return jsonify({'error': 'User not found'}), 404
            
    except Exception as e:
        print(f"DEBUG: Token validation error: {str(e)}")
        return jsonify({'error': 'Token validation failed', 'details': str(e)}), 500

def validate_deriv_token(api_token):
    """Validate Deriv API token"""
    try:
        # This is a mock validation - replace with actual Deriv API validation
        payload = {
            "authorize": api_token
        }
        
        # For now, return success with mock data
        # In production, implement actual Deriv WebSocket API call
        return {
            'success': True,
            'account_id': f"VRTC{api_token[-8:]}",
            'message': 'Token validated successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Token validation failed: {str(e)}'
        }

def get_deriv_balance(api_token):
    """Get balance from Deriv API"""
    try:
        # This is a mock implementation - replace with actual Deriv API call
        # In production, implement WebSocket connection to Deriv API
        
        # Mock balance data
        import random
        mock_balance = round(random.uniform(1000, 10000), 2)
        
        return {
            'success': True,
            'balance': mock_balance,
            'currency': 'USD',
            'message': 'Balance fetched successfully'
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Failed to fetch balance: {str(e)}'
        }
