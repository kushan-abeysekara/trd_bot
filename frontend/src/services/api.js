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
  getDerivsTradeHistory: () => api.get('/trading/history'), // Renamed to avoid conflict
  autoTrade: (settings) => api.post('/trading/auto-trade', settings),
  setupApiToken: (data) => api.post('/deriv/save-token', data),
  getBalance: (accountType) => api.get(`/deriv/balance${accountType ? `?account_type=${accountType}` : ''}`),
  removeApiToken: (accountType) => api.delete(`/deriv/remove-token${accountType ? `?account_type=${accountType}` : ''}`),
  getAccountStatus: (accountType) => api.get(`/deriv/account-status${accountType ? `?account_type=${accountType}` : ''}`),
  switchAccount: (accountType) => api.post('/deriv/switch-account', { account_type: accountType }),
  getStats: (accountType) => api.get(`/trading/stats${accountType ? `?account_type=${accountType}` : ''}`),
  getRecentActivity: (accountType) => api.get(`/trading/activity${accountType ? `?account_type=${accountType}` : ''}`),
  // Add new AI analysis endpoints
  analyzeMarket: (data) => api.post('/ai/analyze-market', data),
  getTradingRecommendation: (symbol, dataPoints) => api.post('/ai/trading-recommendation', { symbol, dataPoints }),
  getMarketPrediction: (data) => api.post('/ai/market-prediction', data),
  // Add new volatility charts endpoints
  getVolatilityData: (symbol, timeframe = '1m', limit = 100) => api.get(`/volatility/data?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`),
  getVolatilityTick: (symbol) => api.get(`/volatility/tick?symbol=${symbol}`),
  subscribeVolatilityTicks: (symbol) => api.post('/volatility/subscribe', { symbol }),
  // Add new real-time analysis endpoints
  getRealtimeMarketData: (symbol) => api.get(`/ai/realtime-market?symbol=${symbol}`),
  predictFutureDigits: (data) => api.post('/ai/predict-digits', data),
  getMarketSentiment: (symbol) => api.get(`/ai/market-sentiment?symbol=${symbol}`),
  getLiveAnalytics: (symbol, timeframe = '1m') => api.get(`/ai/live-analytics?symbol=${symbol}&timeframe=${timeframe}`),
  // Enhanced volatility endpoints
  getRealtimeTicks: (symbol) => api.get(`/volatility/realtime-ticks?symbol=${symbol}`),
  subscribeToMarketStream: (symbol) => api.post('/volatility/stream-subscribe', { symbol }),
  // Advanced AI predictions
  getFutureDigitPredictions: (data) => api.post('/ai/future-digits-prediction', data),
  getRealTimeMarketCondition: (symbol) => api.get(`/ai/market-condition?symbol=${symbol}`),
  getAdvancedTechnicalIndicators: (data) => api.post('/ai/technical-indicators', data),
  // Market Analysis endpoints
  analyzeMarketData: async (data) => {
    const response = await api.post('/market-analysis/analyze', data);
    return response.data;
  },
  
  analyzeMarketRealTime: async (symbol, data) => {
    const response = await api.post(`/market-analysis/real-time/${symbol}`, data);
    return response.data;
  },
  
  getMarketTradingRecommendation: async (data) => {
    const response = await api.post('/market-analysis/trading-recommendation', data);
    return response.data;
  },
  
  getDigitAnalysis: async () => {
    const response = await api.get('/market-analysis/digit-analysis');
    return response.data;
  },
  
  getChatGPTAnalysis: async () => {
    const response = await api.get('/market-analysis/chatgpt-analysis');
    return response.data;
  },
  
  getPredictions: async () => {
    const response = await api.get('/market-analysis/predictions');
    return response.data;
  },
  
  getAnalysisStatus: async () => {
    const response = await api.get('/market-analysis/status');
    return response.data;
  },
  
  // Trading Bot API methods
  getBotStatus: () => api.get('/trading-bot/status'),
  startBot: () => api.post('/trading-bot/start'),
  stopBot: () => api.post('/trading-bot/stop'),
  getActiveTrades: () => api.get('/trading-bot/active-trades'),
  getTradeHistory: async (limit = 50) => {
    try {
      const response = await api.get(`/trading-bot/trade-history?limit=${limit}`);
      // Transform data if needed for frontend compatibility
      if (response.data && response.data.trades) {
        // Ensure proper date formatting and trade status
        response.data.trades = response.data.trades.map(trade => ({
          ...trade,
          entry_time: new Date(trade.entry_time).toISOString(),
          exit_time: trade.exit_time ? new Date(trade.exit_time).toISOString() : null
        }));
      }
      return response;
    } catch (error) {
      console.error('Error fetching trade history:', error);
      throw error;
    }
  },
  getBotStatistics: () => api.get('/trading-bot/statistics'),
  getBotSettings: () => api.get('/trading-bot/settings'),
  updateBotSettings: (settings) => api.put('/trading-bot/settings', settings),
  updateMarketData: (data) => api.post('/trading-bot/update-market-data', data),
  forceCloseTrade: (tradeId) => api.post(`/trading-bot/force-close/${tradeId}`),
  getStrategies: () => api.get('/trading-bot/strategies'),
  getStrategyDetails: (id) => api.get(`/trading-bot/strategies/${id}`),
  setStrategy: (id) => api.post(`/trading-bot/set-strategy/${id}`),
};

export default api;
