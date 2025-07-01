import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';
import { Mail, Phone, Lock, User, Eye, EyeOff, TrendingUp } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import VerificationModal from '../components/VerificationModal';

const RegisterPage = () => {
  const navigate = useNavigate();
  const { register: registerUser } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [registrationType, setRegistrationType] = useState('email');
  const [showVerification, setShowVerification] = useState(false);
  const [registeredUser, setRegisteredUser] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    reset
  } = useForm();

  const password = watch('password');

  const onSubmit = async (data) => {
    setIsLoading(true);
    try {
      // Prepare registration data
      const registrationData = {
        first_name: data.firstName,
        last_name: data.lastName,
        password: data.password
      };

      if (registrationType === 'email') {
        registrationData.email = data.email;
      } else {
        registrationData.mobile_number = data.mobileNumber;
      }

      const response = await registerUser(registrationData);
      
      toast.success('Registration successful! Please verify your account.');
      setRegisteredUser(response.user);
      setShowVerification(true);
      reset();
    } catch (error) {
      toast.error(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerificationSuccess = () => {
    setShowVerification(false);
    toast.success('Account verified successfully! Redirecting to dashboard...');
    setTimeout(() => {
      navigate('/dashboard');
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="max-w-md w-full bg-white rounded-2xl shadow-xl p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <TrendingUp className="w-8 h-8 text-blue-600 mr-2" />
            <h1 className="text-2xl font-bold text-gray-900">TradingBot</h1>
          </div>
          <h2 className="text-xl font-semibold text-gray-700 mb-2">Create Account</h2>
          <p className="text-gray-500">Start your automated trading journey</p>
        </div>

        {/* Registration Type Toggle */}
        <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
          <button
            type="button"
            onClick={() => setRegistrationType('email')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              registrationType === 'email'
                ? 'bg-white text-blue-600 shadow'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Mail className="w-4 h-4 inline mr-2" />
            Email
          </button>
          <button
            type="button"
            onClick={() => setRegistrationType('mobile')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
              registrationType === 'mobile'
                ? 'bg-white text-blue-600 shadow'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            <Phone className="w-4 h-4 inline mr-2" />
            Mobile
          </button>
        </div>

        {/* Registration Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          {/* Name Fields */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                First Name
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  {...register('firstName', {
                    required: 'First name is required',
                    minLength: { value: 2, message: 'At least 2 characters' }
                  })}
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="John"
                />
              </div>
              {errors.firstName && (
                <p className="text-red-500 text-xs mt-1">{errors.firstName.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Last Name
              </label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  {...register('lastName', {
                    required: 'Last name is required',
                    minLength: { value: 2, message: 'At least 2 characters' }
                  })}
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Doe"
                />
              </div>
              {errors.lastName && (
                <p className="text-red-500 text-xs mt-1">{errors.lastName.message}</p>
              )}
            </div>
          </div>

          {/* Email or Mobile Field */}
          {registrationType === 'email' ? (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="email"
                  {...register('email', {
                    required: 'Email is required',
                    pattern: {
                      value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                      message: 'Invalid email address'
                    }
                  })}
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="john.doe@example.com"
                />
              </div>
              {errors.email && (
                <p className="text-red-500 text-xs mt-1">{errors.email.message}</p>
              )}
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Mobile Number
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="tel"
                  {...register('mobileNumber', {
                    required: 'Mobile number is required',
                    pattern: {
                      value: /^[+]?[1-9]\d{1,14}$/,
                      message: 'Invalid mobile number'
                    }
                  })}
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="+94712345678"
                />
              </div>
              {errors.mobileNumber && (
                <p className="text-red-500 text-xs mt-1">{errors.mobileNumber.message}</p>
              )}
            </div>
          )}

          {/* Password Field */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type={showPassword ? 'text' : 'password'}
                {...register('password', {
                  required: 'Password is required',
                  minLength: { value: 8, message: 'At least 8 characters' },
                  pattern: {
                    value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]/,
                    message: 'Password must contain uppercase, lowercase, number and special character'
                  }
                })}
                className="w-full pl-10 pr-10 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="••••••••"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {errors.password && (
              <p className="text-red-500 text-xs mt-1">{errors.password.message}</p>
            )}
          </div>

          {/* Confirm Password Field */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Confirm Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type={showPassword ? 'text' : 'password'}
                {...register('confirmPassword', {
                  required: 'Please confirm your password',
                  validate: value => value === password || 'Passwords do not match'
                })}
                className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="••••••••"
              />
            </div>
            {errors.confirmPassword && (
              <p className="text-red-500 text-xs mt-1">{errors.confirmPassword.message}</p>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        {/* Login Link */}
        <div className="mt-6 text-center">
          <p className="text-gray-600">
            Already have an account?{' '}
            <Link to="/login" className="text-blue-600 hover:text-blue-700 font-medium">
              Sign in
            </Link>
          </p>
        </div>
      </div>

      {/* Verification Modal */}
      {showVerification && registeredUser && (
        <VerificationModal
          isOpen={showVerification}
          onClose={() => setShowVerification(false)}
          user={registeredUser}
          onSuccess={handleVerificationSuccess}
          verificationType={registrationType}
        />
      )}
    </div>
  );
};

export default RegisterPage;
