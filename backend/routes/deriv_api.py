from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json
import os

from models import User, DerivAccount, db
from utils.deriv_service import DerivService

deriv_bp = Blueprint('deriv', __name__, url_prefix='/api/deriv')

# Initialize DerivService with environment variables
class EnhancedDerivService(DerivService):
    def __init__(self):
        super().__init__()
        self.app_id = os.getenv('DERIV_APP_ID', '1089')
        self.ws_url = f"{os.getenv('DERIV_WS_URL', 'wss://ws.binaryws.com/websockets/v3')}?app_id={self.app_id}"

deriv_service = EnhancedDerivService()

@deriv_bp.route('/save-token', methods=['POST'])
@jwt_required()
def save_api_token():
    """Save Deriv API token with enhanced error handling for both Demo and Real"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        print(f"API Token setup attempt for user: {user_id}")
        print(f"Data received: {data}")
        
        if not data.get('api_token'):
            return jsonify({'error': 'API token is required'}), 400
        
        account_type = data.get('account_type', 'demo')
        if account_type not in ['demo', 'real']:
            return jsonify({'error': 'Invalid account type'}), 400
        
        # Get user
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        print(f"Validating {account_type} token for user: {user.first_name} {user.last_name}")
        
        # Validate API token with Deriv
        validation_result = deriv_service.validate_token(data['api_token'])
        print(f"Token validation result: {validation_result}")
        
        if not validation_result['valid']:
            return jsonify({'error': validation_result['message']}), 400
        
        print("Token validation successful, getting account info...")
        
        # Get account info from Deriv
        account_info = deriv_service.get_account_info(data['api_token'])
        print(f"Account info result: {account_info}")
        
        if not account_info['success']:
            return jsonify({'error': f"Failed to get account info: {account_info['message']}"}), 400
        
        # Save API token based on account type
        if account_type == 'demo':
            user.deriv_demo_api_token = data['api_token']
            user.deriv_demo_account_id = account_info['data'].get('loginid')
        else:  # real
            user.deriv_real_api_token = data['api_token']
            user.deriv_real_account_id = account_info['data'].get('loginid')
        
        # Update general fields (for backward compatibility)
        user.deriv_api_token = data['api_token']  # Keep the last configured token
        user.deriv_account_type = account_type
        user.api_token_updated_at = datetime.utcnow()
        user.deriv_account_id = account_info['data'].get('loginid')
        user.deriv_balance = account_info['data'].get('balance', 0.0)
        user.deriv_currency = account_info['data'].get('currency', 'USD')
        
        # Create or update deriv account record
        deriv_account = DerivAccount.query.filter_by(
            user_id=user_id, 
            account_id=user.deriv_account_id
        ).first()
        
        if not deriv_account:
            deriv_account = DerivAccount(
                user_id=user_id,
                account_id=user.deriv_account_id,
                account_type=account_type,
                balance=user.deriv_balance,
                currency=user.deriv_currency
            )
            db.session.add(deriv_account)
        else:
            deriv_account.balance = user.deriv_balance
            deriv_account.account_type = account_type
            deriv_account.last_updated = datetime.utcnow()
        
        db.session.commit()
        
        print(f"{account_type.upper()} API token saved successfully")
        
        return jsonify({
            'message': f'{account_type.upper()} API Token Setup Successfully',
            'account_info': {
                'account_id': user.deriv_account_id,
                'account_type': account_type,
                'balance': user.deriv_balance,
                'currency': user.deriv_currency,
                'has_api_token': True,
                'has_demo_token': bool(user.deriv_demo_api_token),
                'has_real_token': bool(user.deriv_real_api_token)
            }
        }), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"Save API token error: {str(e)}")
        return jsonify({'error': 'Failed to save API token', 'details': str(e)}), 500

@deriv_bp.route('/balance', methods=['GET'])
@jwt_required()
def get_real_time_balance():
    """Get real-time balance from Deriv account"""
    try:
        user_id = get_jwt_identity()
        print(f"Balance request for user ID: {user_id}")
        
        user = User.query.get(user_id)
        
        if not user:
            print(f"User not found for ID: {user_id}")
            return jsonify({'error': 'User not found'}), 404
        
        print(f"User found: {user.first_name} {user.last_name}, has API token: {bool(user.deriv_api_token)}")
        
        if not user or not user.deriv_api_token:
            print("No Deriv API token found for user")
            return jsonify({'error': 'No Deriv API token found'}), 404
        
        # Get real-time balance from Deriv
        print("Attempting to get real-time balance from Deriv")
        balance_result = deriv_service.get_real_time_balance(user.deriv_api_token)
        print(f"Balance result: {balance_result}")
        
        if balance_result['success']:
            # Update user balance in database
            user.deriv_balance = balance_result['data']['balance']
            user.deriv_currency = balance_result['data']['currency']
            
            # Update deriv account record
            if user.deriv_account_id:
                deriv_account = DerivAccount.query.filter_by(
                    user_id=user_id,
                    account_id=user.deriv_account_id
                ).first()
                
                if deriv_account:
                    deriv_account.balance = balance_result['data']['balance']
                    deriv_account.last_updated = datetime.utcnow()
            
            db.session.commit()
            print("Balance updated successfully in database")
            
            return jsonify({
                'balance': balance_result['data']['balance'],
                'currency': balance_result['data']['currency'],
                'account_id': user.deriv_account_id,
                'account_type': user.deriv_account_type,
                'last_updated': datetime.utcnow().isoformat()
            }), 200
        else:
            print(f"Failed to get balance from Deriv: {balance_result['message']}")
            return jsonify({'error': balance_result['message']}), 400
            
    except Exception as e:
        print(f"Get balance error: {str(e)}")
        return jsonify({'error': 'Failed to get balance', 'details': str(e)}), 500

@deriv_bp.route('/account-status', methods=['GET'])
@jwt_required()
def get_account_status():
    """Get Deriv account status and configuration"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        account_status = {
            'api_configured': bool(user.deriv_api_token),
            'account_type': user.deriv_account_type,
            'account_id': user.deriv_account_id,
            'balance': user.deriv_balance,
            'currency': user.deriv_currency,
            'last_updated': user.api_token_updated_at.isoformat() if user.api_token_updated_at else None
        }
        
        return jsonify(account_status), 200
        
    except Exception as e:
        print(f"Get account status error: {str(e)}")
        return jsonify({'error': 'Failed to get account status', 'details': str(e)}), 500

@deriv_bp.route('/remove-token', methods=['DELETE'])
@jwt_required()
def remove_api_token():
    """Remove Deriv API token"""
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Clear API token and related data
        user.deriv_api_token = None
        user.deriv_account_type = 'demo'
        user.deriv_account_id = None
        user.deriv_balance = 0.0
        user.deriv_currency = 'USD'
        user.api_token_updated_at = None
        
        # Deactivate deriv account records
        DerivAccount.query.filter_by(user_id=user_id).update({'is_active': False})
        
        db.session.commit()
        
        return jsonify({'message': 'API token removed successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"Remove API token error: {str(e)}")
        return jsonify({'error': 'Failed to remove API token', 'details': str(e)}), 500

@deriv_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint for deriv routes"""
    return jsonify({'message': 'Deriv API routes working'}), 200
