import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || (
    process.env.NODE_ENV === 'production' 
      ? '/api' 
      : 'http://localhost:5000/api'
  ),
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor to add auth token with better error handling
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url} with token: ${token.substring(0, 20)}...`);
    } else {
      console.log(`API Request: ${config.method?.toUpperCase()} ${config.url} (no token)`);
    }
    return config;
  },
  (error) => {
    console.error('Request interceptor error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.method?.toUpperCase()} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error(`API Error: ${error.response?.status} ${error.config?.method?.toUpperCase()} ${error.config?.url}`, error.response?.data);
    
    // Only redirect to login for 401 errors if we're not already on auth pages
    if (error.response?.status === 401) {
      const currentPath = window.location.pathname;
      const isAuthPage = ['/login', '/register', '/verify-otp'].includes(currentPath);
      
      // Don't redirect if we're already on an auth page or if this is a profile request failing
      if (!isAuthPage && !error.config?.url?.includes('/auth/profile')) {
        console.log('401 error detected, clearing auth data and redirecting to login');
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// Auth API endpoints
export const authAPI = {
  login: (credentials) => api.post('/auth/login', credentials),
  register: (userData) => api.post('/auth/register', userData),
  verify: (verificationData) => api.post('/auth/verify', verificationData),
  resendCode: (data) => api.post('/auth/resend-code', data),
  logout: () => api.post('/auth/logout'),
  profile: () => api.get('/auth/profile')  // Add this endpoint
};

// Trading API endpoints
export const tradingAPI = {
  getProfile: () => api.get('/trading/profile'),
  updateProfile: (data) => api.put('/trading/profile', data),
  getTradeHistory: () => api.get('/trading/history'),
  autoTrade: (settings) => api.post('/trading/auto-trade', settings),
  setupApiToken: (data) => api.post('/deriv/save-token', data),
  getBalance: (accountType) => api.get(`/deriv/balance${accountType ? `?account_type=${accountType}` : ''}`),
  removeApiToken: (accountType) => api.delete(`/deriv/remove-token${accountType ? `?account_type=${accountType}` : ''}`),
  getAccountStatus: (accountType) => api.get(`/deriv/account-status${accountType ? `?account_type=${accountType}` : ''}`),
  switchAccount: (accountType) => api.post('/deriv/switch-account', { account_type: accountType }),
  getStats: (accountType) => api.get(`/trading/stats${accountType ? `?account_type=${accountType}` : ''}`),
  getRecentActivity: (accountType) => api.get(`/trading/activity${accountType ? `?account_type=${accountType}` : ''}`),
  // Add new volatility charts endpoints
  getVolatilityData: (symbol, timeframe = '1m', limit = 100) => api.get(`/volatility/data?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`),
  getVolatilityTick: (symbol) => api.get(`/volatility/tick?symbol=${symbol}`),
  subscribeVolatilityTicks: (symbol) => api.post('/volatility/subscribe', { symbol })
};

export default api;
