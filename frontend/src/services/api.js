import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor to add auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
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
  logout: () => api.post('/auth/logout')
};

// Trading API endpoints (for future use)
export const tradingAPI = {
  getProfile: () => api.get('/trading/profile'),
  updateProfile: (data) => api.put('/trading/profile', data),
  getTradeHistory: () => api.get('/trading/history'),
  autoTrade: (settings) => api.post('/trading/auto-trade', settings)
};

export default api;
