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
  // Add new AI analysis endpoints
  analyzeMarket: (data) => api.post('/ai/analyze-market', data),
  getTradingRecommendation: (symbol, dataPoints) => {
    // Add better validation and debugging
    console.log(`Trading recommendation API call for ${symbol} with ${dataPoints?.length || 0} data points`);
    
    // Ensure data points are properly formatted
    const formattedDataPoints = dataPoints && Array.isArray(dataPoints) 
      ? dataPoints.map(point => {
          if (typeof point === 'number') {
            return { price: point, timestamp: new Date().toISOString() };
          } else if (typeof point === 'object') {
            return {
              price: point.price || point.value || 0,
              timestamp: point.timestamp || new Date().toISOString()
            };
          }
          return { price: 0, timestamp: new Date().toISOString() };
        })
      : [];
      
    return api.post('/ai/trading-recommendation', { 
      symbol, 
      dataPoints: formattedDataPoints, 
      contractType: 'rise_fall' // Default contract type
    });
  },
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
  analyzeMarketAdvanced: async (data) => {
    try {
      const response = await api.post('/market-analysis/analyze', data);
      return response.data;
    } catch (error) {
      console.error('Market analysis error:', error);
      return { error: 'Analysis failed', fallback: true };
    }
  },
  
  analyzeMarketRealTime: async (symbol, data) => {
    try {
      const response = await api.post(`/market-analysis/real-time/${symbol}`, data);
      return response.data;
    } catch (error) {
      console.error('Real-time analysis error:', error);
      return { error: 'Real-time analysis failed', fallback: true };
    }
  },
  
  getTradingRecommendationAdvanced: async (data) => {
    try {
      const response = await api.post('/market-analysis/trading-recommendation', data);
      return response.data;
    } catch (error) {
      console.error('Trading recommendation error:', error);
      return { 
        error: 'Recommendation failed', 
        fallback: true,
        recommendation: {
          contract_type: 'rise_fall',
          direction: 'call',
          confidence: 50,
          risk_level: 'medium',
          duration: '5 minutes',
          reasoning: 'Fallback recommendation due to error'
        }
      };
    }
  },
  
  getDigitAnalysis: async () => {
    try {
      const response = await api.get('/market-analysis/digit-analysis');
      return response.data;
    } catch (error) {
      console.error('Digit analysis error:', error);
      return { error: 'Digit analysis failed', fallback: true };
    }
  },
  
  getChatGPTAnalysis: async () => {
    try {
      const response = await api.get('/market-analysis/chatgpt-analysis');
      return response.data;
    } catch (error) {
      console.error('ChatGPT analysis error:', error);
      return { error: 'ChatGPT analysis failed', fallback: true };
    }
  },
  
  getPredictions: async () => {
    try {
      const response = await api.get('/market-analysis/predictions');
      return response.data;
    } catch (error) {
      console.error('Predictions error:', error);
      return { error: 'Predictions failed', fallback: true };
    }
  },
  
  getAnalysisStatus: async () => {
    try {
      const response = await api.get('/market-analysis/status');
      return response.data;
    } catch (error) {
      console.error('Analysis status error:', error);
      return { status: 'error', fallback: true };
    }
  },
  
  // Trading Bot API endpoints with better error handling
  startTradingBot: async (data) => {
    try {
      const response = await api.post('/trading-bot/start', data);
      return response.data;
    } catch (error) {
      console.error('Start trading bot error:', error);
      throw error;
    }
  },
  
  stopTradingBot: async () => {
    try {
      const response = await api.post('/trading-bot/stop');
      return response.data;
    } catch (error) {
      console.error('Stop trading bot error:', error);
      throw error;
    }
  },
  getBotStatus: () => api.get('/trading-bot/status'),
  getBotSettings: () => api.get('/trading-bot/settings'),
  updateBotSettings: (data) => api.put('/trading-bot/settings', data),
  getTradingHistory: (params) => api.get('/trading-bot/history', { params }),
  getTradingSessions: (params) => api.get('/trading-bot/sessions', { params }),
  getPerformanceAnalytics: () => api.get('/trading-bot/performance'),
  retrainMlModels: () => api.post('/trading-bot/ml-models/retrain'),
  getMlModelPerformance: () => api.get('/trading-bot/ml-models/performance'),
  testTradingSignal: (data) => api.post('/trading-bot/test-signal', data),
  getBotConfigurations: () => api.get('/trading-bot/bot-configs')
};

export default api;
