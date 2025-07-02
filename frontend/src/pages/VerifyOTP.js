import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Mail, Phone, Clock, ArrowLeft } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import toast from 'react-hot-toast';

const VerifyOTP = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { verify, resendVerification } = useAuth();
  
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [isLoading, setIsLoading] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [timer, setTimer] = useState(600); // 10 minutes = 600 seconds
  const [isTimerActive, setIsTimerActive] = useState(true);

  // Get user data from location state
  const user = location.state?.user;
  const verificationType = location.state?.verificationType || 'mobile';

  // Redirect if no user data
  useEffect(() => {
    if (!user) {
      toast.error('No verification data found. Please register again.');
      navigate('/register');
    }
  }, [user, navigate]);

  // Timer effect - 10 minutes countdown
  useEffect(() => {
    let interval = null;
    if (isTimerActive && timer > 0) {
      interval = setInterval(() => {
        setTimer(timer => timer - 1);
      }, 1000);
    } else if (timer === 0) {
      setIsTimerActive(false);
      toast.error('Verification code expired. Please request a new one.');
    }
    return () => clearInterval(interval);
  }, [isTimerActive, timer]);

  // Format timer display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

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

      const result = await verify(verificationData);
      toast.success('Account verified successfully!');
      
      // FIXED: Navigate immediately after successful verification
      navigate('/dashboard', { replace: true });
    } catch (error) {
      toast.error(error.message || 'Verification failed');
      // Reset code on error
      setCode(['', '', '', '', '', '']);
      const firstInput = document.getElementById('code-0');
      if (firstInput) firstInput.focus();
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendCode = async () => {
    setIsResending(true);
    try {
      await resendVerification({ user_id: user.id });
      toast.success('Verification code sent successfully!');
      setTimer(600); // Reset to 10 minutes
      setIsTimerActive(true);
      // Reset code inputs
      setCode(['', '', '', '', '', '']);
      const firstInput = document.getElementById('code-0');
      if (firstInput) firstInput.focus();
    } catch (error) {
      toast.error(error.message || 'Failed to resend code');
    } finally {
      setIsResending(false);
    }
  };

  if (!user) {
    return null; // Will redirect via useEffect
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <button
            onClick={() => navigate('/register')}
            className="inline-flex items-center text-blue-600 hover:text-blue-700 mb-6"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Registration
          </button>
          
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            {verificationType === 'email' ? (
              <Mail className="w-8 h-8 text-blue-600" />
            ) : (
              <Phone className="w-8 h-8 text-blue-600" />
            )}
          </div>
          
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Verify Your Account
          </h2>
          
          <p className="text-gray-600 mb-2">
            We've sent a 6-digit verification code to
          </p>
          <p className="font-medium text-gray-900 mb-6">
            {verificationType === 'email' ? user.email : user.mobile_number}
          </p>
        </div>

        {/* Verification Form */}
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="flex justify-center space-x-2">
            {code.map((digit, index) => (
              <input
                key={index}
                id={`code-${index}`}
                type="text"
                value={digit}
                onChange={(e) => handleCodeChange(index, e.target.value)}
                onKeyDown={(e) => handleKeyDown(index, e)}
                className="w-12 h-12 text-center text-xl font-semibold border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                maxLength={1}
                inputMode="numeric"
                pattern="[0-9]*"
                autoComplete="off"
              />
            ))}
          </div>

          {/* Timer */}
          <div className="text-center">
            {isTimerActive ? (
              <div className="flex items-center justify-center text-gray-500">
                <Clock className="w-4 h-4 mr-2" />
                <span>Code expires in {formatTime(timer)}</span>
              </div>
            ) : (
              <div className="text-red-500">
                <p className="mb-2">Verification code expired</p>
                <button
                  type="button"
                  onClick={handleResendCode}
                  disabled={isResending}
                  className="text-blue-600 hover:text-blue-700 font-medium disabled:opacity-50"
                >
                  {isResending ? 'Sending...' : 'Request new code'}
                </button>
              </div>
            )}
          </div>

          {/* Resend button when timer is active */}
          {isTimerActive && (
            <div className="text-center">
              <p className="text-gray-500 text-sm mb-2">Didn't receive the code?</p>
              <button
                type="button"
                onClick={handleResendCode}
                disabled={isResending}
                className="text-blue-600 hover:text-blue-700 font-medium disabled:opacity-50"
              >
                {isResending ? 'Sending...' : 'Resend code'}
              </button>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading || code.join('').length !== 6 || !isTimerActive}
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
        <div className="text-center text-sm text-gray-500">
          <p>Didn't receive the code?</p>
          <p>Check your messages or contact support if the issue persists.</p>
        </div>
      </div>
    </div>
  );
};

export default VerifyOTP;
