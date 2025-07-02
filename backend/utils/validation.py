import re
import phonenumbers
from email_validator import validate_email, EmailNotValidError

class ValidationService:
    def __init__(self):
        pass
    
    def validate_email_address(self, email):
        """Validate email address"""
        try:
            # Validate and get info about the email
            valid = validate_email(email)
            # The normalized result (accessible via valid.email)
            return True, valid.email
        except EmailNotValidError as e:
            return False, str(e)
    
    def validate_mobile_number(self, mobile_number):
        """Validate mobile number"""
        try:
            # Parse the phone number
            parsed_number = phonenumbers.parse(mobile_number, None)
            
            # Check if the number is valid
            if not phonenumbers.is_valid_number(parsed_number):
                return False, "Invalid mobile number"
            
            # Format the number in international format
            formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
            
            return True, formatted_number
            
        except phonenumbers.NumberParseException as e:
            return False, f"Invalid mobile number: {str(e)}"
    
    def validate_name(self, name):
        """Validate name (first name or last name)"""
        if not name or not name.strip():
            return False, "Name cannot be empty"
        
        name = name.strip()
        
        if len(name) < 2:
            return False, "Name must be at least 2 characters long"
        
        if len(name) > 50:
            return False, "Name must be less than 50 characters"
        
        # Check if name contains only letters, spaces, hyphens, and apostrophes
        if not re.match(r"^[a-zA-Z\s\-']+$", name):
            return False, "Name can only contain letters, spaces, hyphens, and apostrophes"
        
        return True, name
    
    def validate_password(self, password):
        """Validate password strength"""
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must be less than 128 characters"
        
        # Check for at least one lowercase letter
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for at least one uppercase letter
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for at least one digit
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number"
        
        # Check for at least one special character
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is valid"
    
    def validate_verification_code(self, code):
        """Validate verification code"""
        if not code:
            return False, "Verification code cannot be empty"
        
        if not code.isdigit():
            return False, "Verification code must contain only numbers"
        
        if len(code) != 6:
            return False, "Verification code must be 6 digits"
        
        return True, code
