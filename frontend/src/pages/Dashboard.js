import React, { useState, useEffect, useCallback } from 'react';
import { 
  TrendingUp, 
  User, 
  Settings, 
  LogOut, 
  Activity, 
  DollarSign, 
  BarChart3,
  Bot,
  Shield,
  Bell,
  ChevronDown,
  Play,
  Key,
  RefreshCw,
  AlertCircle,
  Target,
  TrendingDown,
  Eye,
  StopCircle,
  ArrowUp,
  ArrowDown
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { tradingAPI } from '../services/api';
import ApiTokenSetup from '../components/ApiTokenSetup';
import VolatilityChart from '../components/VolatilityChart';
import LastDigitDisplay from '../components/LastDigitDisplay';
import AIMarketAnalyzer from '../components/AIMarketAnalyzer';
import StrategySelector from '../components/StrategySelector';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const { user, logout, refreshUser } = useAuth();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [autoTradeEnabled, setAutoTradeEnabled] = useState(user?.auto_trade_enabled || false);
  const [showApiSetup, setShowApiSetup] = useState(false);
  const [accountBalance, setAccountBalance] = useState(null);
  const [isLoadingBalance, setIsLoadingBalance] = useState(false);
  const [balanceError, setBalanceError] = useState(null);
  const [currentAccountType, setCurrentAccountType] = useState(user?.deriv_account_type || 'demo');
  const [stats, setStats] = useState({
    balance: 0,
    profit: 0,
    trades: 0,
    winRate: 0
  });
  const [recentActivity, setRecentActivity] = useState([]);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [isLoadingActivity, setIsLoadingActivity] = useState(false);
  const [lastDigit, setLastDigit] = useState(null);
  const [currentIndexName, setCurrentIndexName] = useState('');
  const [currentPrice, setCurrentPrice] = useState(null);
  const [currentChartData, setCurrentChartData] = useState([]);
  
  // New state variables for enhanced bot functionality
  const [strategyOptions, setStrategyOptions] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [isLoadingStrategies, setIsLoadingStrategies] = useState(false);
  const [latestTradeSignal, setLatestTradeSignal] = useState(null);
  const [tradeSignalHistory, setTradeSignalHistory] = useState([]);
  const [technicalIndicators, setTechnicalIndicators] = useState({
    rsi: 50,
    macd: 0,
    macdSignal: 0,
    momentum: 0,
    volatility: 0,
    bollingerUpper: 0,
    bollingerLower: 0,
    bollingerMiddle: 0,
    ema5: 0
  });
  const [realtimeUpdatesEnabled, setRealtimeUpdatesEnabled] = useState(true);
  const [wsConnection, setWsConnection] = useState(null);

  // Check API token availability for both account types
  const hasDemoToken = user?.has_demo_token || false;
  const hasRealToken = user?.has_real_token || false;
  const hasCurrentAccountToken = currentAccountType === 'demo' ? hasDemoToken : hasRealToken;

  const fetchBalance = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    setIsLoadingBalance(true);
    setBalanceError(null);
    
    try {
      const response = await tradingAPI.getBalance(currentAccountType);
      setAccountBalance(response.data);
    } catch (error) {
      const errorMessage = error.response?.data?.error || 'Failed to fetch balance';
      setBalanceError(errorMessage);
    } finally {
      setIsLoadingBalance(false);
    }
  }, [hasCurrentAccountToken, currentAccountType]);

  const fetchStats = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    setIsLoadingStats(true);
    try {
      const response = await tradingAPI.getStats(currentAccountType);
      setStats(response.data);
    } catch (error) {
      console.error('Failed to fetch stats:', error);
      setStats({ balance: 0, profit: 0, trades: 0, winRate: 0 });
    } finally {
      setIsLoadingStats(false);
    }
  }, [hasCurrentAccountToken, currentAccountType]);

  const fetchRecentActivity = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    setIsLoadingActivity(true);
    try {
      const response = await tradingAPI.getRecentActivity(currentAccountType);
      setRecentActivity(response.data.activities || []);
    } catch (error) {
      console.error('Failed to fetch recent activity:', error);
      setRecentActivity([]);
    } finally {
      setIsLoadingActivity(false);
    }
  }, [hasCurrentAccountToken, currentAccountType]);

  // Fetch balance, stats, and recent activity on component mount and account type change
  useEffect(() => {
    if (hasCurrentAccountToken) {
      fetchBalance();
      fetchStats();
      fetchRecentActivity();
      
      // Set up auto-refresh every 30 seconds
      const interval = setInterval(() => {
        fetchBalance();
        fetchStats();
        fetchRecentActivity();
      }, 30000);
      
      return () => clearInterval(interval);
    } else {
      // Clear data when no token available
      setAccountBalance(null);
      setStats({ balance: 0, profit: 0, trades: 0, winRate: 0 });
      setRecentActivity([]);
    }
  }, [hasCurrentAccountToken, currentAccountType, fetchBalance, fetchStats, fetchRecentActivity]);

  const handleAccountSwitch = async (accountType) => {
    const hasToken = accountType === 'demo' ? hasDemoToken : hasRealToken;
    
    if (!hasToken) {
      toast.error(`Please setup your ${accountType.toUpperCase()} API token first`);
      setShowApiSetup(true);
      return;
    }

    try {
      await tradingAPI.switchAccount(accountType);
      setCurrentAccountType(accountType);
      toast.success(`Switched to ${accountType.toUpperCase()} account`);
    } catch (error) {
      toast.error(`Failed to switch to ${accountType.toUpperCase()} account`);
    }
  };

  // Move API token detection logic before useEffect
  const hasApiToken = user?.has_api_token || user?.api_configured || user?.deriv_api_token;

  // Fetch balance on component mount and set up auto-refresh
  useEffect(() => {
    if (hasApiToken) {
      fetchBalance();
      // Set up auto-refresh every 30 seconds
      const interval = setInterval(fetchBalance, 30000);
      return () => clearInterval(interval);
    }
  }, [hasApiToken, fetchBalance]);

  const handleRemoveApiToken = async () => {
    if (!window.confirm('Are you sure you want to remove your API token? This will disable automated trading.')) {
      return;
    }

    try {
      await tradingAPI.removeApiToken();
      setAccountBalance(null);
      setAutoTradeEnabled(false);
      toast.success('API token removed successfully');
      // Refresh user data or trigger a re-fetch
      window.location.reload();
    } catch (error) {
      toast.error('Failed to remove API token');
    }
  };

  const handleLogout = () => {
    logout();
  };

  const toggleAutoTrade = () => {
    if (!hasApiToken) {
      toast.error('Please setup your API token first');
      setShowApiSetup(true);
      return;
    }
    setAutoTradeEnabled(!autoTradeEnabled);
  };

  const handleApiSetupSuccess = async () => {
    try {
      // Refresh user data from server
      await refreshUser();
      // Refresh balance if API token is now available
      if (hasApiToken) {
        fetchBalance();
      }
      toast.success('Dashboard updated with API configuration');
    } catch (error) {
      console.error('Failed to refresh user data:', error);
      // Don't force reload, just show a warning
      toast.error('API setup successful but failed to refresh dashboard. Please refresh the page.');
    }
  };

  const handleLastDigitUpdate = (digit, indexName, price, chartData) => {
    setLastDigit(digit);
    setCurrentIndexName(indexName);
    setCurrentPrice(price);
    setCurrentChartData(chartData || []);
  };

  // Add bot state management
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    account_balance: 0,
    daily_profit: 0,
    daily_loss: 0,
    win_rate: 0,
    current_strategy: 'ADAPTIVE',
    strategy_status: 'MONITORING',
    active_trades_count: 0,
    settings: {
      auto_stake: 1.0,
      manual_stake: 1.0,
      max_stake: 10.0,
      min_stake: 0.35,
      stake_percentage: 10,
      daily_stop_loss: 50.0,
      daily_target: 20.0,
      max_concurrent_trades: 3
    }
  });
  const [activeTrades, setActiveTrades] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [botStatistics, setBotStatistics] = useState({
    total_trades: 0,
    won_trades: 0,
    lost_trades: 0,
    total_profit: 0,
    total_loss: 0,
    net_profit: 0
  });
  const [isLoadingBot, setIsLoadingBot] = useState(false);
  const [isStartingBot, setIsStartingBot] = useState(false);
  const [isStoppingBot, setIsStoppingBot] = useState(false);

  // Bot API functions
  const fetchBotStatus = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      const response = await tradingAPI.getBotStatus();
      setBotStatus(response.data.status);
    } catch (error) {
      console.error('Failed to fetch bot status:', error);
    }
  }, [hasCurrentAccountToken]);

  const fetchActiveTrades = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      const response = await tradingAPI.getActiveTrades();
      setActiveTrades(response.data.trades || []);
    } catch (error) {
      console.error('Failed to fetch active trades:', error);
      setActiveTrades([]);
    }
  }, [hasCurrentAccountToken]);

  const fetchTradeHistory = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      // Get a larger limit to ensure we have history to display
      const response = await tradingAPI.getTradeHistory(100);
      
      if (response.data && Array.isArray(response.data.trades)) {
        // Process the trades to ensure they have all required fields
        const processedTrades = response.data.trades.map(trade => ({
          ...trade,
          id: trade.id || `trade_${Date.now()}`,
          entry_time: trade.entry_time ? new Date(trade.entry_time).toISOString() : new Date().toISOString(),
          exit_time: trade.exit_time ? new Date(trade.exit_time).toISOString() : null,
          status: trade.status || 'UNKNOWN',
          profit_loss: parseFloat(trade.profit_loss || 0).toFixed(2)
        }));
        setTradeHistory(processedTrades);
        console.log('Trade history loaded:', processedTrades.length, 'trades');
      } else {
        console.warn('Trade history response format is unexpected:', response.data);
        setTradeHistory([]);
      }
    } catch (error) {
      console.error('Failed to fetch trade history:', error);
      toast.error(`Failed to load trade history: ${error.message}`);
      setTradeHistory([]);
    }
  }, [hasCurrentAccountToken]);

  const fetchBotStatistics = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      const response = await tradingAPI.getBotStatistics();
      setBotStatistics(response.data.statistics);
    } catch (error) {
      console.error('Failed to fetch bot statistics:', error);
    }
  }, [hasCurrentAccountToken]);

  // Fetch bot data on mount and account change
  useEffect(() => {
    if (hasCurrentAccountToken) {
      console.log('Initializing bot data fetching');
      
      // Initial data fetch
      fetchBotStatus();
      fetchActiveTrades();
      fetchTradeHistory();
      fetchBotStatistics();
      
      // Set up auto-refresh for real-time data
      const fastRefreshInterval = setInterval(() => {
        fetchBotStatus();
        fetchActiveTrades();
      }, 5000); // Refresh status and active trades every 5 seconds
      
      // Set up slower refresh for less frequently changing data
      const slowRefreshInterval = setInterval(() => {
        fetchTradeHistory();
        fetchBotStatistics();
      }, 15000); // Refresh trade history every 15 seconds
      
      return () => {
        clearInterval(fastRefreshInterval);
        clearInterval(slowRefreshInterval);
      };
    }
  }, [hasCurrentAccountToken, currentAccountType, fetchBotStatus, fetchActiveTrades, fetchTradeHistory, fetchBotStatistics]);

  const handleStartBot = async () => {
    if (!hasCurrentAccountToken) {
      toast.error('Please setup your API token first');
      setShowApiSetup(true);
      return;
    }

    setIsStartingBot(true);
    try {
      // Check current bot status first
      const statusResponse = await tradingAPI.getBotStatus();
      const isAlreadyRunning = statusResponse.data.status?.is_running;
      
      if (isAlreadyRunning) {
        toast.info('Trading bot is already running');
      } else {
        // Start the bot
        const response = await tradingAPI.startBot();
        
        if (response.data && response.data.success) {
          toast.success('Trading bot started successfully');
        } else {
          toast.error(`Failed to start trading bot: ${response.data?.message || 'Unknown error'}`);
        }
      }
      
      // Refresh status regardless
      fetchBotStatus();
      fetchActiveTrades();
      fetchBotStatistics();
    } catch (error) {
      console.error('Error starting bot:', error);
      // Improved error handling - extract more specific error message
      const errorMessage = error.response?.data?.error || error.response?.data?.message || error.message || 'Unknown error';
      toast.error(`Failed to start trading bot: ${errorMessage}`);
    } finally {
      setIsStartingBot(false);
    }
  };

  const handleStopBot = async () => {
    setIsStoppingBot(true);
    try {
      // Check current bot status first
      const statusResponse = await tradingAPI.getBotStatus();
      const isRunning = statusResponse.data.status?.is_running;
      
      if (!isRunning) {
        toast.info('Trading bot is already stopped');
      } else {
        // Stop the bot
        const response = await tradingAPI.stopBot();
        
        if (response.data && response.data.success) {
          toast.success('Trading bot stopped successfully');
        } else {
          toast.error(`Failed to stop trading bot: ${response.data?.message || 'Unknown error'}`);
        }
      }
      
      // Refresh status regardless
      fetchBotStatus();
      fetchActiveTrades();
      fetchBotStatistics();
    } catch (error) {
      console.error('Error stopping bot:', error);
      // Improved error handling - extract more specific error message
      const errorMessage = error.response?.data?.error || error.response?.data?.message || error.message || 'Unknown error';
      toast.error(`Failed to stop trading bot: ${errorMessage}`);
    } finally {
      setIsStoppingBot(false);
    }
  };

  const handleForceCloseTrade = async (tradeId) => {
    if (!window.confirm('Are you sure you want to force close this trade?')) {
      return;
    }

    try {
      await tradingAPI.forceCloseTrade(tradeId);
      toast.success('Trade closed successfully');
      fetchActiveTrades();
      fetchBotStatus();
    } catch (error) {
      toast.error('Failed to close trade');
    }
  };

  // New functions for real-time trading data and strategy selection
  const fetchStrategyOptions = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    setIsLoadingStrategies(true);
    try {
      const response = await tradingAPI.getStrategyOptions();
      setStrategyOptions(response.data.strategies || []);
    } catch (error) {
      console.error('Failed to fetch strategy options:', error);
      setStrategyOptions([]);
    } finally {
      setIsLoadingStrategies(false);
    }
  }, [hasCurrentAccountToken]);

  const handleStrategySelect = (strategy) => {
    setSelectedStrategy(strategy);
    toast.success(`Strategy set to ${strategy.name}`);
  };
  const fetchLatestTradeSignal = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      const response = await tradingAPI.getLatestTradeSignal();
      setLatestTradeSignal(response.data.signal);
    } catch (error) {
      console.error('Failed to fetch latest trade signal:', error);
    }
  }, [hasCurrentAccountToken]);

  const fetchTradeSignalHistory = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      const response = await tradingAPI.getTradeSignalHistory();
      setTradeSignalHistory(response.data.signals || []);
    } catch (error) {
      console.error('Failed to fetch trade signal history:', error);
      setTradeSignalHistory([]);
    }
  }, [hasCurrentAccountToken]);

  const fetchTechnicalIndicators = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    try {
      const response = await tradingAPI.getTechnicalIndicators();
      setTechnicalIndicators(response.data.indicators);
    } catch (error) {
      console.error('Failed to fetch technical indicators:', error);
    }
  }, [hasCurrentAccountToken]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (realtimeUpdatesEnabled && hasCurrentAccountToken) {
      const socket = new WebSocket(`wss://api.example.com/realtime?token=${user.token}`);
      setWsConnection(socket);

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // Handle real-time data updates
        if (data.type === 'trade_signal') {
          setLatestTradeSignal(data.signal);
        } else if (data.type === 'technical_indicator') {
          setTechnicalIndicators((prev) => ({
            ...prev,
            [data.indicator]: data.value
          }));
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      socket.onclose = () => {
        console.log('WebSocket connection closed');
        setWsConnection(null);
      };

      return () => {
        socket.close();
      };
    }
  }, [realtimeUpdatesEnabled, hasCurrentAccountToken, user.token]);

  // Fetch trading strategies
  const fetchStrategies = useCallback(async () => {
    if (!hasCurrentAccountToken) return;
    
    setIsLoadingStrategies(true);
    try {
      const response = await tradingAPI.getStrategies();
      setStrategyOptions(response.data.strategies || []);
      // Set default strategy if one isn't selected yet
      if (response.data.strategies?.length && !selectedStrategy) {
        setSelectedStrategy(response.data.strategies[0]);
      }
    } catch (error) {
      console.error('Failed to fetch strategies:', error);
      toast.error('Failed to fetch trading strategies');
    } finally {
      setIsLoadingStrategies(false);
    }
  }, [hasCurrentAccountToken, selectedStrategy]);

  // Handle strategy selection
  const handleStrategyChange = async (strategy) => {
    if (!strategy || !strategy.id) return;
    
    try {
      await tradingAPI.setStrategy(strategy.id);
      setSelectedStrategy(strategy);
      toast.success(`Strategy set to: ${strategy.name}`);
    } catch (error) {
      console.error('Failed to set strategy:', error);
      toast.error('Failed to set trading strategy');
    }
  };

  // Setup WebSocket connection for real-time updates
  const setupWebSocketConnection = useCallback(() => {
    if (!hasCurrentAccountToken || !realtimeUpdatesEnabled) return;
    
    // Close existing connection if any
    if (wsConnection) {
      wsConnection.close();
    }
    
    // Create new connection
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:5000/ws';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connection established');
      // Send authentication if needed
      if (user?.token) {
        ws.send(JSON.stringify({ type: 'auth', token: user.token }));
      }
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'tick':
            // Update price and technical indicators
            if (data.data.price) {
              setCurrentPrice(data.data.price);
            }
            if (data.data.lastDigit !== undefined) {
              setLastDigit(data.data.lastDigit);
            }
            break;
            
          case 'technical_indicators':
            // Update technical indicators
            setTechnicalIndicators(data.data);
            break;
            
          case 'trade_signal':
            // New trade signal detected
            setLatestTradeSignal(data.data);
            setTradeSignalHistory(prev => [data.data, ...prev.slice(0, 9)]); // Keep last 10
            break;
            
          case 'bot_status':
            // Update bot status
            setBotStatus(prev => ({ ...prev, ...data.data }));
            break;
            
          case 'active_trades':
            // Update active trades
            setActiveTrades(data.data);
            break;
            
          case 'trade_history':
            // Update trade history
            setTradeHistory(data.data);
            break;
            
          case 'bot_statistics':
            // Update bot statistics
            setBotStatistics(data.data);
            break;
            
          default:
            // Handle other message types
            console.log('Received message:', data);
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
      // Attempt to reconnect after delay
      setTimeout(() => {
        if (realtimeUpdatesEnabled) {
          setupWebSocketConnection();
        }
      }, 5000);
    };
    
    setWsConnection(ws);
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [hasCurrentAccountToken, realtimeUpdatesEnabled, wsConnection, user]);
  
  // Initialize WebSocket connection when component mounts
  useEffect(() => {
    setupWebSocketConnection();
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, [setupWebSocketConnection]);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Header */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <div className="flex items-center">
              <TrendingUp className="w-8 h-8 text-blue-600 mr-2" />
              <h1 className="text-xl font-bold text-gray-900">TradingBot</h1>
            </div>

            {/* Current Price and Last Digit Display - Desktop */}
            <div className="hidden md:flex items-center space-x-4">
              {/* Fixed width container to prevent layout shift */}
              {/* Current Price Section - Fixed Height */}
              <div className="w-48 h-12 flex items-center">
                {currentPrice ? (
                  <div className="flex items-center space-x-3 px-3 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100 w-full h-full">
                    <div className="text-right flex-1">
                      <p className="text-xs text-blue-600 font-medium">Current Price</p>
                      <p className="text-sm font-bold text-blue-900 transition-all duration-300 font-mono min-w-[60px]">
                        {currentPrice.toFixed(2)}
                      </p>
                    </div>
                    <div className="h-8 w-px bg-blue-200"></div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center px-3 py-2 bg-gray-50 rounded-lg border border-gray-100 w-full h-full">
                    <div className="animate-pulse">
                      <div className="h-3 bg-gray-300 rounded w-16 mb-1"></div>
                      <div className="h-4 bg-gray-300 rounded w-12"></div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Fixed width for LastDigitDisplay */}
              <div className="w-auto min-w-[200px]">
                <LastDigitDisplay lastDigit={lastDigit} indexName={currentIndexName} />
              </div>
            </div>

            {/* Account Type Toggle and User Menu */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-600">Account:</span>
                <div className="flex bg-gray-100 rounded-lg p-1">
                  <button
                    onClick={() => handleAccountSwitch('demo')}
                    disabled={!hasDemoToken}
                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                      currentAccountType === 'demo'
                        ? 'bg-blue-600 text-white'
                        : hasDemoToken
                        ? 'text-gray-600 hover:text-gray-900'
                        : 'text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    Demo
                    {!hasDemoToken && <span className="ml-1 text-xs">⚠️</span>}
                  </button>
                  <button
                    onClick={() => handleAccountSwitch('real')}
                    disabled={!hasRealToken}
                    className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                      currentAccountType === 'real'
                        ? 'bg-red-600 text-white'
                        : hasRealToken
                        ? 'text-gray-600 hover:text-gray-900'
                        : 'text-gray-400 cursor-not-allowed'
                    }`}
                  >
                    Real
                    {!hasRealToken && <span className="ml-1 text-xs">⚠️</span>}
                  </button>
                </div>
              </div>

              {/* User Menu */}
              <div className="relative">
                <button
                  onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                  className="flex items-center space-x-2 text-gray-700 hover:text-gray-900 focus:outline-none"
                >
                  <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4 text-blue-600" />
                  </div>
                  <span className="font-medium">{user?.first_name} {user?.last_name}</span>
                  <ChevronDown className="w-4 h-4" />
                </button>

                {/* Dropdown Menu */}
                {isDropdownOpen && (
                  <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10">
                    <div className="px-4 py-2 text-sm text-gray-500 border-b">
                      {user?.email || user?.mobile_number}
                    </div>
                    <button className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                      <Settings className="w-4 h-4 mr-2" />
                      Settings
                    </button>
                    <button 
                      onClick={handleLogout}
                      className="flex items-center w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                    >
                      <LogOut className="w-4 h-4 mr-2" />
                      Logout
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Mobile Last Digit Display */}
        <div className="md:hidden mb-6">
          <div className="flex justify-center">
            <div className="w-full max-w-md">
              {/* Fixed height container to prevent layout shift */}
              <div className="h-20 flex items-center justify-center mb-3">
                {currentPrice ? (
                  <div className="flex items-center space-x-4 px-4 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl shadow-sm w-full">
                    <div className="text-center flex-1">
                      <p className="text-xs text-blue-600 font-medium mb-1">Current Price</p>
                      <p className="text-lg font-bold text-blue-900 transition-all duration-300 font-mono min-w-[80px]">
                        {currentPrice.toFixed(2)}
                      </p>
                    </div>
                    <div className="h-12 w-px bg-blue-200"></div>
                    <div className="text-center flex-1">
                      <p className="text-xs text-blue-600 font-medium mb-1">Last Digit</p>
                      <p className="text-lg font-bold text-blue-900 transition-all duration-300 font-mono min-w-[20px]">
                        {lastDigit ?? '-'}
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl w-full">
                    <div className="text-center">
                      <div className="animate-pulse">
                        <div className="h-4 bg-gray-300 rounded w-20 mb-2 mx-auto"></div>
                        <div className="h-6 bg-gray-300 rounded w-16 mx-auto"></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              
              {/* Fixed height for LastDigitDisplay */}
              <div className="h-12 flex justify-center items-center">
                <LastDigitDisplay lastDigit={lastDigit} indexName={currentIndexName} />
              </div>
            </div>
          </div>
        </div>

        {/* API Token Warning */}
        {!hasCurrentAccountToken && (
          <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <AlertCircle className="w-6 h-6 text-amber-600" />
                <div>
                  <h3 className="font-medium text-amber-800">
                    {currentAccountType.toUpperCase()} API Token Required
                  </h3>
                  <p className="text-sm text-amber-700">
                    Please configure your {currentAccountType} API token to view account data and enable trading.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowApiSetup(true)}
                className="bg-amber-600 text-white px-4 py-2 rounded-lg hover:bg-amber-700 transition-colors font-medium"
              >
                Setup Now
              </button>
            </div>
          </div>
        )}

        {/* Stats Cards */}
        {/* Stats Cards Grid - Fixed Heights */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Balance Card - Fixed Height */}
          <div className="bg-white rounded-lg shadow-md border border-gray-100 h-44">
            <div className="p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-600 truncate">
                    {currentAccountType.charAt(0).toUpperCase() + currentAccountType.slice(1)} Balance
                  </p>
                  <div className="h-12 flex items-center">
                    {isLoadingBalance ? (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                        <span className="text-lg font-bold text-gray-400 font-mono min-w-[100px]">Loading...</span>
                      </div>
                    ) : accountBalance ? (
                      <p className="text-2xl font-bold text-blue-600 truncate font-mono min-w-[120px]">
                        {accountBalance.currency} {accountBalance.balance?.toFixed(2) || '0.00'}
                      </p>
                    ) : hasCurrentAccountToken ? (
                      <p className="text-2xl font-bold text-gray-400 font-mono min-w-[60px]">N/A</p>
                    ) : (
                      <p className="text-xl font-bold text-gray-400 font-mono min-w-[80px]">No Token</p>
                    )}
                  </div>
                </div>
                <div className="flex flex-col items-center space-y-2 ml-4">
                  <div className={`w-12 h-12 rounded-lg flex items-center justify-center transition-colors ${
                    accountBalance ? 'bg-green-100' : 'bg-gray-100'
                  }`}>
                    <DollarSign className={`w-6 h-6 transition-colors ${
                      accountBalance ? 'text-green-600' : 'text-gray-400'
                    }`} />
                  </div>
                  {hasCurrentAccountToken && (
                    <button
                      onClick={fetchBalance}
                      disabled={isLoadingBalance}
                      className="p-1 text-blue-600 hover:text-blue-700 disabled:opacity-50 transition-opacity"
                      title="Refresh balance"
                    >
                      <RefreshCw className={`w-3 h-3 ${isLoadingBalance ? 'animate-spin' : ''}`} />
                    </button>
                  )}
                </div>
              </div>
              
              <div className="mt-auto space-y-2">
                {/* Fixed height for account info */}
                <div className="h-4">
                  {accountBalance?.account_id && (
                    <p className="text-xs text-gray-500 truncate">
                      Account: {accountBalance.account_id}
                    </p>
                  )}
                </div>
                
                {/* Fixed height for status/error */}
                <div className="h-6 flex items-center justify-between">
                  {balanceError ? (
                    <span className="text-xs text-red-600 truncate flex-1">{balanceError}</span>
                  ) : accountBalance ? (
                    <>
                      <span className="text-xs text-gray-500 truncate flex-1">
                        {new Date(accountBalance.last_updated).toLocaleTimeString()}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ml-2 ${
                        accountBalance.account_type === 'demo' 
                          ? 'bg-blue-100 text-blue-800' 
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {accountBalance.account_type === 'demo' ? 'Demo' : 'Real'}
                      </span>
                    </>
                  ) : null}
                </div>
              </div>
            </div>
          </div>

          {/* Total Profit Card - Fixed Height */}
          <div className="bg-white rounded-lg shadow-md border border-gray-100 h-44">
            <div className="p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-600">Total Profit</p>
                  <div className="h-12 flex items-center">
                    {isLoadingStats ? (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-green-600"></div>
                        <span className="text-lg font-bold text-gray-400 font-mono min-w-[80px]">Loading...</span>
                      </div>
                    ) : hasCurrentAccountToken ? (
                      <p className={`text-2xl font-bold transition-colors font-mono min-w-[100px] ${
                        stats.profit > 0 ? 'text-green-600' : stats.profit < 0 ? 'text-red-600' : 'text-gray-600'
                      }`}>
                        ${stats.profit?.toFixed(2) || '0.00'}
                      </p>
                    ) : (
                      <p className="text-xl font-bold text-gray-400 font-mono min-w-[80px]">No Data</p>
                    )}
                  </div>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center ml-4">
                  <TrendingUp className="w-6 h-6 text-green-600" />
                </div>
              </div>
              
              <div className="mt-auto">
                <div className="h-6 flex items-center">
                  {hasCurrentAccountToken && stats.profit !== 0 && (
                    <span className="text-xs text-gray-500">
                      {stats.profit > 0 ? '↗️ Profitable' : '↘️ In Loss'}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Total Trades Card - Fixed Height */}
          <div className="bg-white rounded-lg shadow-md border border-gray-100 h-44">
            <div className="p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-600">Total Trades</p>
                  <div className="h-12 flex items-center">
                    {isLoadingStats ? (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-600"></div>
                        <span className="text-lg font-bold text-gray-400 font-mono min-w-[60px]">Loading...</span>
                      </div>
                    ) : hasCurrentAccountToken ? (
                      <p className="text-2xl font-bold text-gray-900 font-mono min-w-[40px]">
                        {stats.trades || 0}
                      </p>
                    ) : (
                      <p className="text-xl font-bold text-gray-400 font-mono min-w-[60px]">No Data</p>
                    )}
                  </div>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center ml-4">
                  <BarChart3 className="w-6 h-6 text-purple-600" />
                </div>
              </div>
              
              <div className="mt-auto">
                <div className="h-6 flex items-center">
                  {hasCurrentAccountToken && stats.trades > 0 && (
                    <span className="text-xs text-gray-500">
                      📊 Total executed
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Win Rate Card - Fixed Height */}
          <div className="bg-white rounded-lg shadow-md border border-gray-100 h-44">
            <div className="p-6 h-full flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-600">Win Rate</p>
                  <div className="h-12 flex items-center">
                    {isLoadingStats ? (
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-yellow-600"></div>
                        <span className="text-lg font-bold text-gray-400 font-mono min-w-[60px]">Loading...</span>
                      </div>
                    ) : hasCurrentAccountToken ? (
                      <p className="text-2xl font-bold text-gray-900 font-mono min-w-[50px]">
                        {stats.winRate?.toFixed(1) || '0.0'}%
                      </p>
                    ) : (
                      <p className="text-xl font-bold text-gray-400 font-mono min-w-[60px]">No Data</p>
                    )}
                  </div>
                </div>
                <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center ml-4">
                  <Activity className="w-6 h-6 text-yellow-600" />
                </div>
              </div>
              
              <div className="mt-auto">
                <div className="h-6 flex items-center">
                  {hasCurrentAccountToken && stats.winRate > 0 && (
                    <span className="text-xs text-gray-500">
                      {stats.winRate >= 50 ? '🎯 Good performance' : '⚠️ Needs improvement'}
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Auto Trading Control and Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Volatility Chart */}
            <VolatilityChart onLastDigitUpdate={handleLastDigitUpdate} />
            
            {/* AI Market Analyzer */}
            <AIMarketAnalyzer 
              chartData={currentChartData}
              currentPrice={currentPrice}
              selectedIndex={currentIndexName}
            />
            
            {/* Enhanced Trading Bot Control Panel */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                {/* Header with bot status and controls */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                  <div className="flex items-center space-x-3">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                      botStatus.is_running ? 'bg-green-100' : 'bg-gray-100'
                    }`}>
                      <Bot className={`w-6 h-6 ${
                        botStatus.is_running ? 'text-green-600' : 'text-gray-500'
                      }`} />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">Rise/Fall Trading Bot</h3>
                      <div className="flex items-center text-sm">
                        <span className={`font-medium ${
                          botStatus.is_running ? 'text-green-600' : 'text-gray-500'
                        }`}>
                          {botStatus.is_running ? 'Active' : 'Inactive'}
                        </span>
                        <div className={`w-2 h-2 rounded-full ml-2 ${
                          botStatus.is_running ? 'bg-green-500 animate-pulse' : 'bg-gray-300'
                        }`}></div>
                        {botStatus.is_running && (
                          <span className="ml-3 text-gray-500">
                            Status: {botStatus.strategy_status}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    {!botStatus.is_running ? (
                      <button
                        onClick={handleStartBot}
                        disabled={isStartingBot || !hasCurrentAccountToken}
                        className="flex items-center space-x-2 px-5 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-lg hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 transition-all shadow-sm"
                      >
                        {isStartingBot ? (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        ) : (
                          <Play className="w-4 h-4" />
                        )}
                        <span className="font-medium">{isStartingBot ? 'Starting...' : 'Start Bot'}</span>
                      </button>
                    ) : (
                      <button
                        onClick={handleStopBot}
                        disabled={isStoppingBot}
                        className="flex items-center space-x-2 px-5 py-2 bg-gradient-to-r from-red-500 to-rose-600 text-white rounded-lg hover:from-red-600 hover:to-rose-700 disabled:opacity-50 transition-all shadow-sm"
                      >
                        {isStoppingBot ? (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                        ) : (
                          <StopCircle className="w-4 h-4" />
                        )}
                        <span className="font-medium">{isStoppingBot ? 'Stopping...' : 'Stop Bot'}</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Strategy selector */}
                <div className="mb-6">
                  <StrategySelector 
                    initialStrategies={strategyOptions}
                    initialSelectedStrategy={selectedStrategy}
                    onStrategyChange={handleStrategyChange}
                  />
                </div>
                
                {/* Bot Status Cards with enhanced visualization */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200 transition-all hover:shadow-md">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-blue-600 font-medium">Current Strategy</p>
                        <p className="text-lg font-bold text-blue-900">{botStatus.current_strategy || "Not set"}</p>
                        <div className="flex items-center mt-1">
                          <div className={`w-2 h-2 rounded-full ${botStatus.is_running ? 'bg-green-500' : 'bg-gray-300'}`}></div>
                          <p className="text-xs text-blue-700 ml-1.5">{botStatus.strategy_status}</p>
                        </div>
                      </div>
                      <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                        <Target className="w-6 h-6 text-blue-600" />
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200 transition-all hover:shadow-md">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-green-600 font-medium">Daily P&L</p>
                        <p className={`text-lg font-bold ${
                          (botStatus.daily_profit - botStatus.daily_loss) > 0 ? 'text-green-700' : 'text-red-600'
                        }`}>
                          ${(botStatus.daily_profit - botStatus.daily_loss).toFixed(2)}
                        </p>
                        <div className="flex items-center gap-3 text-xs mt-1">
                          <span className="text-green-700">
                            +${botStatus.daily_profit.toFixed(2)}
                          </span>
                          <span className="text-red-600">
                            -${botStatus.daily_loss.toFixed(2)}
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-col items-center">
                        <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                          <TrendingUp className="w-6 h-6 text-green-600" />
                        </div>
                        <div className="mt-1 text-xs font-medium text-gray-500">{botStatistics.total_trades} trades</div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-gradient-to-r from-purple-50 to-violet-50 rounded-lg border border-purple-200 transition-all hover:shadow-md">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-purple-600 font-medium">Performance</p>
                        <div className="flex items-center">
                          <p className="text-lg font-bold text-purple-900">{botStatus.win_rate.toFixed(1)}%</p>
                          <span className="ml-2 text-sm text-gray-600">win rate</span>
                        </div>
                        <div className="flex items-center gap-3 text-xs mt-1">
                          <span className="text-green-600">Won: {botStatistics.won_trades}</span>
                          <span className="text-red-600">Lost: {botStatistics.lost_trades}</span>
                        </div>
                      </div>
                      <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center">
                        <Activity className="w-6 h-6 text-purple-600" />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Active trading panel with real-time data */}
                <div className="bg-gray-50 border rounded-lg p-4 mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium text-gray-900">Real-time Trading Data</h4>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full ${realtimeUpdatesEnabled ? 'bg-green-500 animate-pulse' : 'bg-gray-300'} mr-2`}></div>
                      <span className="text-xs text-gray-600">{realtimeUpdatesEnabled ? 'Live Updates' : 'Updates Paused'}</span>
                    </div>
                  </div>
                  
                  {/* Technical indicators display */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                    <div className="p-2 bg-white rounded border shadow-sm">
                      <p className="text-xs text-gray-500">RSI</p>
                      <div className="flex items-center">
                        <p className={`text-sm font-bold ${
                          technicalIndicators.rsi > 70 ? 'text-red-600' :
                          technicalIndicators.rsi < 30 ? 'text-green-600' : 'text-gray-900'
                        }`}>
                          {technicalIndicators.rsi.toFixed(1)}
                        </p>
                        <div className="ml-auto w-12 h-4 bg-gray-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${
                              technicalIndicators.rsi > 70 ? 'bg-red-500' :
                              technicalIndicators.rsi < 30 ? 'bg-green-500' : 'bg-blue-500'
                            }`}
                            style={{width: `${technicalIndicators.rsi}%`}}
                          ></div>
                        </div>
                      </div>
                    </div>
                    <div className="p-2 bg-white rounded border shadow-sm">
                      <p className="text-xs text-gray-500">MACD</p>
                      <div className="flex items-center">
                        <p className={`text-sm font-bold ${
                          technicalIndicators.macd > 0.2 ? 'text-green-600' :
                          technicalIndicators.macd < -0.2 ? 'text-red-600' : 'text-gray-900'
                        }`}>
                          {technicalIndicators.macd.toFixed(3)}
                        </p>
                        <div className="ml-auto w-12 h-4 bg-gray-100 rounded-full overflow-hidden">
                          <div 
                            className={`h-full ${technicalIndicators.macd > 0 ? 'bg-green-500' : 'bg-red-500'}`}
                            style={{
                              width: `${Math.min(Math.abs(technicalIndicators.macd * 100), 100)}%`,
                              marginLeft: technicalIndicators.macd < 0 ? 'auto' : '0'
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    <div className="p-2 bg-white rounded border shadow-sm">
                      <p className="text-xs text-gray-500">Momentum</p>
                      <p className={`text-sm font-bold ${
                        technicalIndicators.momentum > 0.001 ? 'text-green-600' :
                        technicalIndicators.momentum < -0.001 ? 'text-red-600' : 'text-gray-900'
                      }`}>
                        {(technicalIndicators.momentum * 100).toFixed(3)}%
                      </p>
                    </div>
                    <div className="p-2 bg-white rounded border shadow-sm">
                      <p className="text-xs text-gray-500">Volatility</p>
                      <p className={`text-sm font-bold ${
                        technicalIndicators.volatility > 0.015 ? 'text-amber-600' :
                        technicalIndicators.volatility < 0.005 ? 'text-blue-600' : 'text-gray-900'
                      }`}>
                        {(technicalIndicators.volatility * 100).toFixed(2)}%
                      </p>
                    </div>
                  </div>

                  {/* Latest trading signal */}
                  {latestTradeSignal && (
                    <div className={`p-3 rounded-lg border ${
                      latestTradeSignal.direction === 'RISE' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                    }`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center">
                          {latestTradeSignal.direction === 'RISE' ? (
                            <ArrowUp className="w-4 h-4 text-green-600 mr-2" />
                          ) : (
                            <ArrowDown className="w-4 h-4 text-red-600 mr-2" />
                          )}
                          <span className={`text-sm font-medium ${
                            latestTradeSignal.direction === 'RISE' ? 'text-green-700' : 'text-red-700'
                          }`}>
                            Latest Signal: {latestTradeSignal.direction}
                          </span>
                        </div>
                        <div className="flex items-center text-xs">
                          <span className="text-gray-500 mr-2">Confidence:</span>
                          <span className="font-medium">{latestTradeSignal.confidence}%</span>
                        </div>
                      </div>
                      <p className="text-xs text-gray-600 mt-1">{latestTradeSignal.reason}</p>
                    </div>
                  )}
                </div>

                {/* Bot Settings Display */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="p-3 bg-white rounded-lg border shadow-sm hover:shadow-md transition-all">
                    <p className="text-xs text-gray-600 mb-1 flex items-center">
                      <DollarSign className="w-3 h-3 mr-1" />
                      Current Stake
                    </p>
                    <p className="font-bold text-gray-900">${botStatus.settings.auto_stake}</p>
                  </div>
                  <div className="p-3 bg-white rounded-lg border shadow-sm hover:shadow-md transition-all">
                    <p className="text-xs text-gray-600 mb-1 flex items-center">
                      <Activity className="w-3 h-3 mr-1" />
                      Active Trades
                    </p>
                    <p className="font-bold text-gray-900">{botStatus.active_trades_count}/{botStatus.settings.max_concurrent_trades}</p>
                  </div>
                  <div className="p-3 bg-white rounded-lg border shadow-sm hover:shadow-md transition-all">
                    <p className="text-xs text-gray-600 mb-1 flex items-center">
                      <TrendingUp className="w-3 h-3 mr-1" />
                      Daily Target
                    </p>
                    <p className="font-bold text-green-600">${botStatus.settings.daily_target}</p>
                  </div>
                  <div className="p-3 bg-white rounded-lg border shadow-sm hover:shadow-md transition-all">
                    <p className="text-xs text-gray-600 mb-1 flex items-center">
                      <TrendingDown className="w-3 h-3 mr-1" />
                      Stop Loss
                    </p>
                    <p className="font-bold text-red-600">${botStatus.settings.daily_stop_loss}</p>
                  </div>
                </div>

                {/* Active Trades Section */}
                {activeTrades.length > 0 ? (
                  <div className="border-t pt-6">
                    <h4 className="text-md font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                      <Eye className="w-5 h-5 text-blue-600" />
                      <span>Active Trades ({activeTrades.length})</span>
                    </h4>
                    <div className="space-y-3">
                      {activeTrades.map((trade, index) => (
                        <div 
                          key={trade.id || index} 
                          className={`p-4 rounded-lg border shadow-sm transition-all hover:shadow-md ${
                            trade.direction === 'RISE' 
                              ? 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-200' 
                              : 'bg-gradient-to-r from-red-50 to-rose-50 border-red-200'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center mb-2">
                                {trade.direction === 'RISE' ? (
                                  <ArrowUp className="w-4 h-4 text-green-600 mr-2" />
                                ) : (
                                  <ArrowDown className="w-4 h-4 text-red-600 mr-2" />
                                )}
                                <p className="font-medium text-gray-900">
                                  {trade.direction} - {trade.symbol}
                                </p>
                              </div>
                              <div className="grid grid-cols-3 gap-4 text-sm">
                                <div>
                                  <p className="text-xs text-gray-500">Stake</p>
                                  <p className="font-medium">${trade.stake}</p>
                                </div>
                                <div>
                                  <p className="text-xs text-gray-500">Entry Price</p>
                                  <p className="font-medium">${trade.entry_price}</p>
                                </div>
                                <div>
                                  <p className="text-xs text-gray-500">Strategy</p>
                                  <p className="font-medium truncate" title={trade.strategy}>{trade.strategy.split(':')[0]}</p>
                                </div>
                              </div>
                            </div>
                            <div className="flex flex-col items-end ml-4">
                              <div className="flex items-center mb-2">
                                <p className="text-xs bg-gray-200 text-gray-800 rounded px-2 py-1">
                                  {trade.duration}s
                                </p>
                                <div className={`ml-2 w-2 h-2 rounded-full ${
                                  trade.status === 'ACTIVE' ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
                                }`}></div>
                              </div>
                              <button
                                onClick={() => handleForceCloseTrade(trade.id)}
                                className="px-3 py-1 text-xs bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                              >
                                Force Close
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : botStatus.is_running ? (
                  <div className="border-t pt-6 text-center">
                    <div className="py-8 flex flex-col items-center text-gray-500">
                      <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                        <Activity className="w-8 h-8 text-gray-300" />
                      </div>
                      <p className="text-lg font-medium mb-1">No Active Trades</p>
                      <p className="text-sm">Waiting for trading signals...</p>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
                  <button
                    onClick={fetchRecentActivity}
                    disabled={isLoadingActivity || !hasCurrentAccountToken}
                    className="p-2 text-blue-600 hover:text-blue-700 disabled:opacity-50 hover:bg-blue-50 rounded-lg transition-colors"
                    title="Refresh activity"
                  >
                    <RefreshCw className={`w-4 h-4 ${isLoadingActivity ? 'animate-spin' : ''}`} />
                  </button>
                </div>
                
                {/* Fixed height container to prevent layout shifts */}
                <div className="min-h-[300px]">
                  {!hasCurrentAccountToken ? (
                    <div className="flex flex-col items-center justify-center h-[300px] text-gray-500">
                      <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                        <Activity className="w-8 h-8 text-gray-300" />
                      </div>
                      <p className="text-lg font-medium mb-2">No API Token</p>
                      <p className="text-sm text-center">Configure API token to view recent activity</p>
                    </div>
                  ) : isLoadingActivity ? (
                    <div className="flex flex-col items-center justify-center h-[300px]">
                      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mb-4"></div>
                      <p className="text-gray-500">Loading activities...</p>
                    </div>
                  ) : recentActivity.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-[300px] text-gray-500">
                      <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                        <Activity className="w-8 h-8 text-gray-300" />
                      </div>
                      <p className="text-lg font-medium mb-2">No Recent Activities</p>
                      <p className="text-sm text-center">Your trading activities will appear here</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {recentActivity.map((activity, index) => (
                        <div 
                          key={activity.id || index} 
                          className="flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border border-gray-200 hover:shadow-sm transition-all duration-200"
                        >
                          <div className="flex items-center space-x-3 flex-1 min-w-0">
                            <div className={`w-3 h-3 rounded-full flex-shrink-0 ${
                              activity.type === 'profit' ? 'bg-green-400' : 
                              activity.type === 'loss' ? 'bg-red-400' : 'bg-blue-400'
                            }`}></div>
                            <div className="min-w-0 flex-1">
                              <p className="font-medium text-gray-900 truncate">{activity.title}</p>
                              <p className="text-sm text-gray-600 truncate">{activity.description}</p>
                            </div>
                          </div>
                          <div className="text-right flex-shrink-0 ml-4">
                            <p className={`font-bold ${
                              activity.amount > 0 ? 'text-green-600' : 
                              activity.amount < 0 ? 'text-red-600' : 'text-gray-600'
                            }`}>
                              {activity.amount > 0 ? '+' : ''}${Math.abs(activity.amount).toFixed(2)}
                            </p>
                            <p className="text-xs text-gray-500 whitespace-nowrap">{activity.time}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Bot Statistics */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-blue-600" />
                  <span>Bot Performance</span>
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center p-3 bg-green-50 border border-green-200 rounded-lg">
                    <span className="text-gray-700 font-medium">Net Profit</span>
                    <span className={`font-bold ${
                      botStatistics.net_profit > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ${botStatistics.net_profit?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <span className="text-gray-700 font-medium">Total Trades</span>
                    <span className="text-blue-600 font-bold">{botStatistics.total_trades || 0}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-purple-50 border border-purple-200 rounded-lg">
                    <span className="text-gray-700 font-medium">Win Rate</span>
                    <span className="text-purple-600 font-bold">
                      {botStatistics.total_trades > 0 
                        ? ((botStatistics.won_trades / botStatistics.total_trades) * 100).toFixed(1)
                        : '0.0'
                      }%
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="p-2 bg-green-50 border border-green-200 rounded text-center">
                      <p className="text-xs text-green-600">Won</p>
                      <p className="font-bold text-green-700">{botStatistics.won_trades || 0}</p>
                    </div>
                    <div className="p-2 bg-red-50 border border-red-200 rounded text-center">
                      <p className="text-xs text-red-600">Lost</p>
                      <p className="font-bold text-red-700">{botStatistics.lost_trades || 0}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* API Configuration */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">API Configuration</h3>
                {hasApiToken ? (
                  <div className="space-y-4">
                    {/* Overall Status - Fixed Height */}
                    <div className="h-20 p-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg">
                      <div className="flex items-center justify-between h-full">
                        <div className="flex items-center space-x-3">
                          <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                            <Shield className="w-5 h-5 text-green-600" />
                          </div>
                          <div>
                            <span className="font-medium text-green-800">API Configured</span>
                            <p className="text-sm text-green-700">
                              Ready for trading
                            </p>
                          </div>
                        </div>
                        <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                      </div>
                    </div>

                    {/* API Status Details - Fixed Heights */}
                    <div className="space-y-3">
                      {/* Demo Account Status */}
                      <div className="h-12 flex items-center justify-between p-3 border border-gray-200 rounded-lg bg-gray-50">
                        <div className="flex items-center space-x-3">
                          <span className="text-gray-700 font-medium">Demo API</span>
                          {hasDemoToken ? (
                            <div className="flex items-center space-x-1">
                              <Shield className="w-4 h-4 text-green-600" />
                              <span className="text-xs text-green-600 font-medium">Active</span>
                            </div>
                          ) : (
                            <span className="text-xs text-gray-400">Not Set</span>
                          )}
                        </div>
                        {currentAccountType === 'demo' && (
                          <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded-full">Current</span>
                        )}
                      </div>

                      {/* Real Account Status */}
                      <div className="h-12 flex items-center justify-between p-3 border border-gray-200 rounded-lg bg-gray-50">
                        <div className="flex items-center space-x-3">
                          <span className="text-gray-700 font-medium">Real API</span>
                          {hasRealToken ? (
                            <div className="flex items-center space-x-1">
                              <Shield className="w-4 h-4 text-green-600" />
                              <span className="text-xs text-green-600 font-medium">Active</span>
                            </div>
                          ) : (
                            <span className="text-xs text-gray-400">Not Set</span>
                          )}
                        </div>
                        {currentAccountType === 'real' && (
                          <span className="text-xs text-red-600 bg-red-100 px-2 py-1 rounded-full">Current</span>
                        )}
                      </div>

                      {/* Account ID - Fixed Height */}
                      {accountBalance?.account_id && (
                        <div className="h-12 flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
                          <span className="text-gray-700 font-medium">Account ID</span>
                          <span className="text-sm text-gray-900 font-mono bg-white px-2 py-1 rounded border">
                            {accountBalance.account_id}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Action Buttons - Fixed Height */}
                    <div className="flex space-x-2 pt-2">
                      <button
                        onClick={() => setShowApiSetup(true)}
                        className="flex-1 h-10 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-lg px-3 transition-colors flex items-center justify-center"
                      >
                        Manage APIs
                      </button>
                      <button
                        onClick={handleRemoveApiToken}
                        className="flex-1 h-10 text-red-600 hover:text-red-700 text-sm font-medium border border-red-200 hover:border-red-300 rounded-lg px-3 transition-colors flex items-center justify-center"
                      >
                        Remove API
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center">
                    <div className="h-20 flex flex-col items-center justify-center mb-4">
                      <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mb-2">
                        <Key className="w-6 h-6 text-green-600" />
                      </div>
                    </div>
                    <div className="h-16 p-3 bg-green-50 border border-green-200 rounded-lg mb-4">
                      <p className="text-green-700 font-medium mb-2">Ready to Configure</p>
                      <p className="text-sm text-green-600">
                        Setup your Deriv API tokens to enable automated trading
                      </p>
                    </div>
                    <button
                      onClick={() => setShowApiSetup(true)}
                      className="w-full h-10 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors font-medium"
                    >
                      Setup API Tokens
                    </button>
                    <p className="text-xs text-gray-500 mt-2 h-4">
                      You can configure both Demo and Real API tokens
                    </p>
                  </div>
                )}
              </div>
            </div>

            {/* Account Status */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Status</h3>
                <div className="space-y-4">
                  <div className="h-10 flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
                    <span className="text-gray-700 font-medium">Verification</span>
                    <div className="flex items-center space-x-2">
                      <Shield className="w-4 h-4 text-green-600" />
                      <span className="text-green-600 font-medium">Verified</span>
                    </div>
                  </div>
                  <div className="h-10 flex items-center justify-between p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <span className="text-gray-700 font-medium">Plan</span>
                    <span className="text-blue-600 font-bold">Premium</span>
                  </div>
                  <div className="h-10 flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
                    <span className="text-gray-700 font-medium">API Status</span>
                    <span className="text-green-600 font-medium">Connected</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
                <div className="space-y-2">
                  <button className="w-full h-12 text-left p-3 text-gray-700 hover:bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg transition-all duration-200 border border-transparent hover:border-blue-200">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                        <DollarSign className="w-4 h-4 text-blue-600" />
                      </div>
                      <span className="font-medium">Add Funds</span>
                    </div>
                  </button>
                  <button className="w-full h-12 text-left p-3 text-gray-700 hover:bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg transition-all duration-200 border border-transparent hover:border-green-200">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                        <BarChart3 className="w-4 h-4 text-green-600" />
                      </div>
                      <span className="font-medium">View Reports</span>
                    </div>
                  </button>
                  <button className="w-full h-12 text-left p-3 text-gray-700 hover:bg-gradient-to-r from-gray-50 to-slate-50 rounded-lg transition-all duration-200 border border-transparent hover:border-gray-200">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                        <Settings className="w-4 h-4 text-gray-600" />
                      </div>
                      <span className="font-medium">Trading Settings</span>
                    </div>
                  </button>
                  <button className="w-full h-12 text-left p-3 text-gray-700 hover:bg-gradient-to-r from-purple-50 to-violet-50 rounded-lg transition-all duration-200 border border-transparent hover:border-purple-200">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                        <Bell className="w-4 h-4 text-purple-600" />
                      </div>
                      <span className="font-medium">Notifications</span>
                    </div>
                  </button>
                </div>
              </div>
            </div>

            {/* Deriv Connection Status */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Deriv Integration</h3>
                <div className="text-center">
                  {user?.deriv_account_id ? (
                    <div className="h-32 flex flex-col items-center justify-center">
                      <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-3">
                        <Shield className="w-8 h-8 text-green-600" />
                      </div>
                      <p className="text-green-600 font-bold text-lg">Connected</p>
                      <p className="text-sm text-gray-600 mt-1 font-mono bg-gray-50 px-2 py-1 rounded">
                        {user.deriv_account_id}
                      </p>
                    </div>
                  ) : (
                    <div className="h-32 flex flex-col items-center justify-center">
                      <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mb-3">
                        <Shield className="w-8 h-8 text-orange-600" />
                      </div>
                      <p className="text-orange-600 font-bold text-lg mb-3">Not Connected</p>
                      <button className="w-full h-10 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
                        Connect Deriv Account
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* API Token Setup Modal */}
      <ApiTokenSetup
        isOpen={showApiSetup}
        onClose={() => setShowApiSetup(false)}
        onSuccess={handleApiSetupSuccess}
        user={user}
        currentAccountType={currentAccountType}
      />
    </div>
  );
};

export default Dashboard;
