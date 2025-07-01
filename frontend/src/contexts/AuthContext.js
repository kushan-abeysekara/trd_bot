import React, { createContext, useContext, useReducer, useEffect } from 'react';
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
      const response = await authAPI.login(credentials);
      dispatch({
        type: 'LOGIN_SUCCESS',
        payload: response.data
      });
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Login failed';
      dispatch({
        type: 'LOGIN_FAILURE',
        payload: errorMessage
      });
      throw new Error(errorMessage);
    }
  };

  // Register function
  const register = async (userData) => {
    dispatch({ type: 'LOGIN_START' });
    try {
      const response = await authAPI.register(userData);
      return response.data;
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Registration failed';
      dispatch({
        type: 'LOGIN_FAILURE',
        payload: errorMessage
      });
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

  const value = {
    ...state,
    login,
    register,
    verify,
    logout
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
