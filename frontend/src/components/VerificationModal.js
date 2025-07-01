import React, { useState, useEffect } from 'react';
import { X, Mail, Phone, Clock } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const VerificationModal = ({ 
  isOpen, 
  onClose, 
  user, 
  onSuccess, 
  verificationType = 'email' 
}) => {
  const { verify, resendVerification } = useAuth();
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [isLoading, setIsLoading] = useState(false);
  const [timer, setTimer] = useState(60);
  const [isTimerActive, setIsTimerActive] = useState(true);

  // Timer effect
  useEffect(() => {
    let interval = null;
    if (isTimerActive && timer > 0) {
      interval = setInterval(() => {
        setTimer(timer => timer - 1);
      }, 1000);
    } else if (timer === 0) {
      setIsTimerActive(false);
    }
    return () => clearInterval(interval);
  }, [isTimerActive, timer]);

  const handleCodeChange = (index, value) => {
    if (value.length > 1) return;
    
    const newCode = [...code];
    newCode[index] = value;
    setCode(newCode);

    // Auto-focus next input
    if (value && index < 5) {
      const nextInput = document.getElementById(`code-${index + 1}`);
      if (nextInput) nextInput.focus();
    }
  };

  const handleKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !code[index] && index > 0) {
      const prevInput = document.getElementById(`code-${index - 1}`);
      if (prevInput) prevInput.focus();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const verificationCode = code.join('');
    
    if (verificationCode.length !== 6) {
      toast.error('Please enter the complete verification code');
      return;
    }

    setIsLoading(true);
    try {
      const verificationData = {
        user_id: user.id,
        verification_code: verificationCode
      };

      if (verificationType === 'email') {
        verificationData.email = user.email;
      } else {
        verificationData.mobile_number = user.mobile_number;
      }

      await verify(verificationData);
      onSuccess();
    } catch (error) {
      toast.error(error.message);
      // Reset code on error
      setCode(['', '', '', '', '', '']);
      const firstInput = document.getElementById('code-0');
      if (firstInput) firstInput.focus();
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendCode = async () => {
    if (!user?.id) {
      toast.error('User information not available');
      return;
    }

    try {
      await resendVerification({ user_id: user.id });
      toast.success('Verification code sent!');
      setTimer(60);
      setIsTimerActive(true);
      // Reset the code inputs
      setCode(['', '', '', '', '', '']);
      const firstInput = document.getElementById('code-0');
      if (firstInput) firstInput.focus();
    } catch (error) {
      toast.error('Failed to resend code');
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">Verify Your Account</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="text-center mb-6">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            {verificationType === 'email' ? (
              <Mail className="w-8 h-8 text-blue-600" />
            ) : (
              <Phone className="w-8 h-8 text-blue-600" />
            )}
          </div>
          <p className="text-gray-600 mb-2">
            We've sent a 6-digit verification code to
          </p>
          <p className="font-medium text-gray-900">
            {verificationType === 'email' ? user?.email : user?.mobile_number}
          </p>
        </div>

        {/* Verification Form */}
        <form onSubmit={handleSubmit}>
          <div className="flex justify-center space-x-2 mb-6">
            {code.map((digit, index) => (
              <input
                key={index}
                id={`code-${index}`}
                type="text"
                value={digit}
                onChange={(e) => handleCodeChange(index, e.target.value)}
                onKeyDown={(e) => handleKeyDown(index, e)}
                className="w-12 h-12 text-center text-xl font-semibold border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                maxLength={1}
                inputMode="numeric"
                pattern="[0-9]*"
              />
            ))}
          </div>

          {/* Timer */}
          <div className="text-center mb-6">
            {isTimerActive ? (
              <div className="flex items-center justify-center text-gray-500">
                <Clock className="w-4 h-4 mr-2" />
                <span>Resend code in {timer}s</span>
              </div>
            ) : (
              <button
                type="button"
                onClick={handleResendCode}
                className="text-blue-600 hover:text-blue-700 font-medium"
              >
                Resend verification code
              </button>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading || code.join('').length !== 6}
            className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Verifying...
              </div>
            ) : (
              'Verify Account'
            )}
          </button>
        </form>

        {/* Help Text */}
        <div className="mt-4 text-center text-sm text-gray-500">
          Didn't receive the code? Check your spam folder or try resending.
        </div>
      </div>
    </div>
  );
};

export default VerificationModal;

