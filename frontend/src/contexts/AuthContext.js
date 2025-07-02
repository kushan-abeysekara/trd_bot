import { createContext, useContext, useReducer, useEffect } from 'react';
import { authAPI } from '../services/api';

// Auth Context
const AuthContext = createContext();

// Auth Reducer
const authReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN_START':
      return { ...state, loading: true, error: null };
    case 'LOGIN_SUCCESS':
      localStorage.setItem('token', action.payload.token);
      localStorage.setItem('user', JSON.stringify(action.payload.user));
      return {
        ...state,
        loading: false,
        user: action.payload.user,
        token: action.payload.token,
        error: null
      };
    case 'LOGIN_FAILURE':
      return { ...state, loading: false, error: action.payload };
    case 'LOGOUT':
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      return { ...state, user: null, token: null, loading: false };
    case 'LOAD_USER':
      return {
        ...state,
        user: action.payload.user,
        token: action.payload.token,
        loading: false
      };
    case 'SET_LOADING':
      return { ...state, loading: action.payload };
    default:
      return state;
  }
};

// Initial state
const initialState = {
  user: null,
  token: null,
  loading: true,
  error: null
};

// Auth Provider
export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Load user from localStorage on app start
  useEffect(() => {
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');
    
    if (token && user) {
      try {
        dispatch({
          type: 'LOAD_USER',
          payload: {
            token,
            user: JSON.parse(user)
          }
        });
      } catch (error) {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    } else {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);

  // Login function
  const login = async (credentials) => {
    dispatch({ type: 'LOGIN_START' });
    try {
      console.log('Attempting login with credentials:', { ...credentials, password: '***' });
      
      const response = await authAPI.login(credentials);
      console.log('Login response received:', { 
        message: response.data.message, 
        hasToken: !!response.data.token,
        hasUser: !!response.data.user 
      });
      
      // FIXED: Better validation of response data
      const { token, access_token, user } = response.data;
      const authToken = token || access_token;
      
      if (!authToken || !user) {
        console.error('Login response validation failed:', {
          hasToken: !!authToken,
          hasUser: !!user,
          tokenType: typeof authToken,
          userType: typeof user
        });
        throw new Error('Invalid login response: missing authentication data');
      }
      
      // Prepare the payload with the correct token
      const loginPayload = {
        token: authToken,
        user: user
      };
      
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: loginPayload
      });
      
      console.log('Login successful, token stored in localStorage');
      return response.data;
    } catch (error) {
      console.error('Login error:', error);
      const errorMessage = error.response?.data?.error || error.message || 'Login failed';
      dispatch({
        type: 'LOGIN_FAILURE',
        payload: errorMessage
      });
      
      // Enhanced error handling for verification
      if (error.response?.data?.verification_required && error.response?.data?.user_id) {
        const enhancedError = new Error(errorMessage);
        enhancedError.user_id = error.response.data.user_id;
        throw enhancedError;
      }
      
      throw new Error(errorMessage);
    }
  };

  // Register function
  const register = async (userData) => {
    try {
      const response = await authAPI.register(userData);
      // Don't dispatch LOGIN_START or LOGIN_FAILURE for registration
      // Registration doesn't log the user in immediately
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Registration failed';
      throw new Error(errorMessage);
    }
  };

  // Verify function
  const verify = async (verificationData) => {
    try {
      const response = await authAPI.verify(verificationData);
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: response.data
      });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Verification failed';
      throw new Error(errorMessage);
    }
  };

  // Logout function
  const logout = () => {
    dispatch({ type: 'LOGOUT' });
  };

  // Resend verification function
  const resendVerification = async (userData) => {
    try {
      const response = await authAPI.resendCode(userData);
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Failed to resend verification code';
      throw new Error(errorMessage);
    }
  };

  // Refresh user data function
  const refreshUser = async () => {
    try {
      const response = await authAPI.profile();
      const updatedUser = response.data.user;
      
      // Update localStorage
      localStorage.setItem('user', JSON.stringify(updatedUser));
      
      // Update state
      dispatch({
        type: 'LOAD_USER',
        payload: {
          token: state.token,
          user: updatedUser
        }
      });
      
      return updatedUser;
    } catch (error) {
      console.error('Failed to refresh user data:', error);
      throw error;
    }
  };

  const value = {
    ...state,
    login,
    register,
    verify,
    logout,
    resendVerification,
    refreshUser
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
