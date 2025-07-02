from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta
import random
import string

from models import User, UserSession, db
from utils.sms_service import SMSService
from utils.validation import ValidationService

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
sms_service = SMSService()
validation_service = ValidationService()

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user with email or mobile number"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['first_name', 'last_name', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.replace("_", " ").title()} is required', 'code': f'MISSING_{field.upper()}'}), 400
        
        # Check if email or mobile is provided
        email = data.get('email')
        mobile_number = data.get('mobile_number')
        
        if not email and not mobile_number:
            return jsonify({'error': 'Either email or mobile number is required'}), 400
        
        if email and mobile_number:
            return jsonify({'error': 'Please provide either email or mobile number, not both'}), 400
        
        # Validate input data
        first_name_valid, first_name_msg = validation_service.validate_name(data['first_name'])
        if not first_name_valid:
            return jsonify({'error': f'First name: {first_name_msg}'}), 400
        
        last_name_valid, last_name_msg = validation_service.validate_name(data['last_name'])
        if not last_name_valid:
            return jsonify({'error': f'Last name: {last_name_msg}'}), 400
        
        password_valid, password_msg = validation_service.validate_password(data['password'])
        if not password_valid:
            return jsonify({'error': password_msg}), 400
        
        # Validate email if provided
        if email:
            email_valid, email_msg = validation_service.validate_email_address(email)
            if not email_valid:
                return jsonify({'error': email_msg}), 400
            email = email_msg  # Use the normalized email
            
            # Check if email already exists
            if User.query.filter_by(email=email).first():
                return jsonify({'error': 'Email already registered'}), 409
        
        # Validate mobile number if provided
        if mobile_number:
            mobile_valid, mobile_msg = validation_service.validate_mobile_number(mobile_number)
            if not mobile_valid:
                return jsonify({'error': mobile_msg}), 400
            mobile_number = mobile_msg  # Use the normalized mobile number
            
            # Check if mobile number already exists
            if User.query.filter_by(mobile_number=mobile_number).first():
                return jsonify({'error': 'Mobile number already registered'}), 409
        
        # Generate verification code
        verification_code = sms_service.generate_verification_code()
        verification_expires = datetime.utcnow() + timedelta(minutes=10)
        
        # Create new user
        user = User(
            email=email,
            mobile_number=mobile_number,
            first_name=first_name_msg,
            last_name=last_name_msg,
            verification_code=verification_code,
            verification_code_expires=verification_expires,
            is_verified=False
        )
        user.set_password(data['password'])
        
        # Save user to database
        db.session.add(user)
        db.session.commit()
        
        # Send verification code - don't fail registration if SMS fails
        sms_sent = False
        sms_error_message = None
        if mobile_number:
            try:
                print(f"Attempting to send SMS to: {mobile_number}")
                sms_result = sms_service.send_verification_sms(mobile_number, verification_code)
                print(f"SMS Result: {sms_result}")
                
                if sms_result['success']:
                    sms_sent = True
                    print("SMS sent successfully!")
                else:
                    sms_error_message = sms_result['message']
                    print(f"SMS sending failed: {sms_result['message']}")
            except Exception as sms_error:
                sms_error_message = str(sms_error)
                print(f"SMS sending error: {str(sms_error)}")
        
        response_data = {
            'message': 'Registration successful. Please verify your account.',
            'user': {
                'id': user.id,
                'email': user.email,
                'mobile_number': user.mobile_number,
                'first_name': user.first_name,
                'last_name': user.last_name
            },
            'verification_required': True,
            'verification_method': 'mobile' if mobile_number else 'email',
            'sms_sent': sms_sent
        }
        
        # Add SMS error info for debugging (only in development)
        if not sms_sent and sms_error_message:
            response_data['sms_error'] = sms_error_message
        
        return jsonify(response_data), 201
        
    except Exception as e:
        db.session.rollback()
        print(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed', 'details': str(e)}), 500

@auth_bp.route('/verify', methods=['POST'])
def verify_account():
    """Verify user account with verification code"""
    try:
        data = request.get_json()
        
        if not data.get('user_id') or not data.get('verification_code'):
            return jsonify({'error': 'User ID and verification code are required'}), 400
        
        # Find user
        user = User.query.get(data['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Check if already verified
        if user.is_verified:
            return jsonify({'error': 'Account already verified'}), 400
        
        # Check verification code
        if user.verification_code != data['verification_code']:
            return jsonify({'error': 'Invalid verification code'}), 400
        
        # Check if code expired
        if user.verification_code_expires < datetime.utcnow():
            return jsonify({'error': 'Verification code expired'}), 400
        
        # Verify user
        user.is_verified = True
        user.verification_code = None
        user.verification_code_expires = None
        db.session.commit()
        
        # Send welcome SMS
        if user.mobile_number:
            sms_service.send_welcome_sms(user.mobile_number, user.first_name)
        
        # Create access token
        access_token = create_access_token(identity=str(user.id))  # Convert to string for PyJWT compatibility
        
        return jsonify({
            'message': 'Account verified successfully',
            'token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Verification failed', 'details': str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user with email/mobile and password"""
    try:
        data = request.get_json()
        
        # Debug logging
        print(f"Login attempt - Data received: {data}")
        
        # Validate input data
        if not data:
            print("Login failed: No data provided")
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email')
        mobile_number = data.get('mobile_number')
        password = data.get('password')
        
        print(f"Login attempt - Email: {email}, Mobile: {mobile_number}, Password: {'*' * len(password) if password else 'None'}")
        
        if not password:
            print("Login failed: Password is required")
            return jsonify({'error': 'Password is required'}), 400
        
        if not email and not mobile_number:
            print("Login failed: Email or mobile number is required")
            return jsonify({'error': 'Email or mobile number is required'}), 400
        
        # Ensure database connection is working
        if not User.ensure_database_connection():
            print("Login failed: Database connection error")
            return jsonify({'error': 'Database connection error. Please try again.'}), 503
        
        # Find user by email or mobile number with enhanced error handling
        user = None
        try:
            if email:
                print(f"Searching for user by email: {email}")
                user = User.query.filter_by(email=email).first()
            else:
                print(f"Searching for user by mobile: {mobile_number}")
                user = User.query.filter_by(mobile_number=mobile_number).first()
                
            print(f"User found: {user is not None}")
            if user:
                print(f"User details - ID: {user.id}, Email: {user.email}, Mobile: {user.mobile_number}, Verified: {user.is_verified}, Active: {user.is_active}")
                
        except Exception as db_error:
            print(f"Database query error: {str(db_error)}")
            error_msg = str(db_error).lower()
            
            # Handle different types of database errors
            if any(phrase in error_msg for phrase in ["unknown column", "no such column", "doesn't exist"]):
                try:
                    print("Attempting to fix database schema...")
                    User.migrate_schema()
                    # Retry the query after migration
                    if email:
                        user = User.query.filter_by(email=email).first()
                    else:
                        user = User.query.filter_by(mobile_number=mobile_number).first()
                    print("Database schema fixed and query retried successfully")
                except Exception as retry_error:
                    print(f"Retry query error: {str(retry_error)}")
                    return jsonify({'error': 'Database configuration error. Please contact support.'}), 500
            elif "connection" in error_msg or "timeout" in error_msg:
                return jsonify({'error': 'Database connection timeout. Please try again.'}), 503
            else:
                return jsonify({'error': 'Database error occurred. Please try again.'}), 500
        
        if not user:
            print("Login failed: User not found")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check password
        password_valid = user.check_password(password)
        print(f"Password validation result: {password_valid}")
        if not password_valid:
            print("Login failed: Invalid password")
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check if user is active
        if not user.is_active:
            print("Login failed: Account is deactivated")
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Check if user is verified
        if not user.is_verified:
            print("Login failed: Account not verified")
            return jsonify({
                'error': 'Account not verified',
                'user_id': user.id,
                'verification_required': True
            }), 401
        
        print("Login validation passed, creating token...")
        
        # Create access token with longer expiration to prevent logout during API setup
        access_token = create_access_token(
            identity=str(user.id),  # Convert to string for PyJWT compatibility
            expires_delta=timedelta(hours=24)  # 24 hour token to prevent logout
        )
        
        print(f"Token created successfully: {access_token[:20]}...")
        
        # Create user session with better error handling
        try:
            session = UserSession(
                user_id=user.id,
                token=access_token,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )
            db.session.add(session)
            db.session.commit()
            print("User session created successfully")
        except Exception as session_error:
            print(f"Session creation error: {str(session_error)}")
            # Continue even if session creation fails, but log the error
            try:
                db.session.rollback()
            except:
                pass
        
        # Get user data safely including API configuration - FIXED FORMAT
        try:
            user_data = user.to_dict()
            # Add API configuration status with multiple field names for compatibility
            user_data.update({
                'has_api_token': bool(user.deriv_api_token),
                'api_configured': bool(user.deriv_api_token),
                'deriv_account_type': user.deriv_account_type,
                'deriv_balance': user.deriv_balance,
                'deriv_currency': user.deriv_currency,
                'deriv_account_id': user.deriv_account_id
            })
            print(f"User data prepared successfully for user ID: {user.id}")
        except Exception as user_dict_error:
            print(f"User dict conversion error: {str(user_dict_error)}")
            # Fallback to basic user data
            user_data = {
                'id': user.id,
                'email': user.email,
                'mobile_number': user.mobile_number,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'is_verified': user.is_verified,
                'has_api_token': bool(getattr(user, 'deriv_api_token', None)),
                'api_configured': bool(getattr(user, 'deriv_api_token', None)),
                'deriv_account_type': getattr(user, 'deriv_account_type', 'demo'),
                'deriv_balance': getattr(user, 'deriv_balance', 0.0),
                'deriv_currency': getattr(user, 'deriv_currency', 'USD'),
                'deriv_account_id': getattr(user, 'deriv_account_id', None)
            }
        
        # FIXED: Ensure both token and user are present and valid
        if not access_token or not user_data:
            print(f"Login response validation failed - Token: {bool(access_token)}, User: {bool(user_data)}")
            return jsonify({'error': 'Failed to create login session'}), 500
        
        response_data = {
            'message': 'Login successful',
            'token': access_token,
            'access_token': access_token,  # Ensure both fields are present
            'user': user_data
        }
        
        print(f"Login response prepared successfully for user: {user.first_name} {user.last_name}")
        print(f"Response contains - Token: {bool(response_data.get('token'))}, User: {bool(response_data.get('user'))}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        try:
            db.session.rollback()
        except:
            pass
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed. Please try again.', 'details': str(e)}), 500

@auth_bp.route('/resend-code', methods=['POST'])  # Add this endpoint
def resend_code():
    """Alias for resend-verification for frontend compatibility"""
    return resend_verification()

@auth_bp.route('/resend-verification', methods=['POST'])
def resend_verification():
    """Resend verification code"""
    try:
        data = request.get_json()
        
        if not data.get('user_id'):
            return jsonify({'error': 'User ID is required'}), 400
        
        user = User.query.get(data['user_id'])
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if user.is_verified:
            return jsonify({'error': 'Account already verified'}), 400
        
        # Generate new verification code
        verification_code = sms_service.generate_verification_code()
        verification_expires = datetime.utcnow() + timedelta(minutes=10)
        
        user.verification_code = verification_code
        user.verification_code_expires = verification_expires
        db.session.commit()
        
        # Send verification code - don't fail if SMS fails
        sms_sent = False
        sms_error_message = None
        if user.mobile_number:
            try:
                print(f"Resending SMS to: {user.mobile_number}")
                sms_result = sms_service.send_verification_sms(user.mobile_number, verification_code)
                print(f"Resend SMS Result: {sms_result}")
                
                if sms_result['success']:
                    sms_sent = True
                    print("SMS resent successfully!")
                else:
                    sms_error_message = sms_result['message']
                    print(f"SMS resending failed: {sms_result['message']}")
            except Exception as sms_error:
                sms_error_message = str(sms_error)
                print(f"SMS resending error: {str(sms_error)}")
        
        response_data = {
            'message': 'Verification code sent successfully' if sms_sent else 'Verification code generated but SMS delivery may have failed',
            'sms_sent': sms_sent
        }
        
        # Add SMS error info for debugging (only in development)
        if not sms_sent and sms_error_message:
            response_data['sms_error'] = sms_error_message
        
        return jsonify(response_data), 200
        
    except Exception as e:
        db.session.rollback()
        print(f"Resend verification error: {str(e)}")
        return jsonify({'error': 'Failed to resend verification code', 'details': str(e)}), 500

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile with enhanced error handling"""
    try:
        user_id = get_jwt_identity()
        
        # Ensure database connection
        if not User.ensure_database_connection():
            return jsonify({'error': 'Database connection error'}), 503
        
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get user data safely
        try:
            user_data = user.to_dict()
        except Exception as user_dict_error:
            print(f"Profile dict conversion error: {str(user_dict_error)}")
            # Try to migrate schema and retry
            try:
                User.migrate_schema()
                user_data = user.to_dict()
            except Exception as retry_error:
                print(f"Profile retry error: {str(retry_error)}")
                # Return basic user data as fallback
                user_data = {
                    'id': user.id,
                    'email': user.email,
                    'mobile_number': user.mobile_number,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'is_verified': user.is_verified,
                    'has_api_token': bool(getattr(user, 'deriv_api_token', None)),
                    'api_configured': bool(getattr(user, 'deriv_api_token', None)),
                    'deriv_account_type': getattr(user, 'deriv_account_type', 'demo'),
                    'deriv_balance': getattr(user, 'deriv_balance', 0.0),
                    'deriv_currency': getattr(user, 'deriv_currency', 'USD'),
                    'deriv_account_id': getattr(user, 'deriv_account_id', None)
                }
        
        return jsonify({'user': user_data}), 200
        
    except Exception as e:
        print(f"Profile error: {str(e)}")
        return jsonify({'error': 'Failed to get profile', 'details': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user"""
    try:
        user_id = get_jwt_identity()
        
        # Deactivate user sessions
        UserSession.query.filter_by(user_id=user_id, is_active=True).update({'is_active': False})
        db.session.commit()
        
        return jsonify({'message': 'Logged out successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': 'Logout failed', 'details': str(e)}), 500

@auth_bp.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Auth routes working'}), 200
