import re
import phonenumbers
from email_validator import validate_email, EmailNotValidError

class ValidationService:
    @staticmethod
    def validate_email_address(email):
        """Validate email address format"""
        try:
            # Validate and get info about the email
            valid = validate_email(email)
            return True, valid.email
        except EmailNotValidError:
            return False, "Invalid email address format"
    
    @staticmethod
    def validate_mobile_number(mobile_number):
        """Validate mobile number format"""
        try:
            # Remove any spaces or special characters
            cleaned_number = re.sub(r'[^\d+]', '', mobile_number)
            
            # Parse the phone number
            parsed_number = phonenumbers.parse(cleaned_number, None)
            
            # Check if the number is valid
            if phonenumbers.is_valid_number(parsed_number):
                # Format the number in international format
                formatted_number = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.E164)
                return True, formatted_number
            else:
                return False, "Invalid mobile number format"
                
        except phonenumbers.NumberParseException:
            return False, "Invalid mobile number format"
    
    @staticmethod
    def validate_password(password):
        """Validate password strength"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one number"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is valid"
    
    @staticmethod
    def validate_name(name):
        """Validate name format"""
        if not name or len(name.strip()) < 2:
            return False, "Name must be at least 2 characters long"
        
        if not re.match(r'^[a-zA-Z\s]+$', name.strip()):
            return False, "Name can only contain letters and spaces"
        
        return True, name.strip()
