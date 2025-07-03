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
  Pause,
  Key,
  RefreshCw,
  AlertCircle,
  Power,
  Target,
  Clock,
  TrendingDown,
  Zap,
  Eye,
  StopCircle
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { tradingAPI } from '../services/api';
import ApiTokenSetup from '../components/ApiTokenSetup';
import VolatilityChart from '../components/VolatilityChart';
import LastDigitDisplay from '../components/LastDigitDisplay';
import AIMarketAnalyzer from '../components/AIMarketAnalyzer';
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
      const response = await tradingAPI.getTradeHistory();
      setTradeHistory(response.data.trades || []);
    } catch (error) {
      console.error('Failed to fetch trade history:', error);
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
      fetchBotStatus();
      fetchActiveTrades();
      fetchTradeHistory();
      fetchBotStatistics();
      
      // Set up auto-refresh for bot data
      const botInterval = setInterval(() => {
        fetchBotStatus();
        fetchActiveTrades();
        fetchBotStatistics();
      }, 5000); // Refresh every 5 seconds for real-time updates
      
      return () => clearInterval(botInterval);
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
      await tradingAPI.startBot();
      toast.success('Trading bot started successfully');
      fetchBotStatus();
    } catch (error) {
      toast.error('Failed to start trading bot');
    } finally {
      setIsStartingBot(false);
    }
  };

  const handleStopBot = async () => {
    setIsStoppingBot(true);
    try {
      await tradingAPI.stopBot();
      toast.success('Trading bot stopped successfully');
      fetchBotStatus();
    } catch (error) {
      toast.error('Failed to stop trading bot');
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
            
            {/* Trading Bot Control Panel */}
            <div className="bg-white rounded-lg shadow-md border border-gray-100">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                    <Bot className="w-6 h-6 text-blue-600" />
                    <span>AI Trading Bot</span>
                  </h3>
                  <div className="flex items-center space-x-3">
                    <div className="flex items-center space-x-2">
                      <span className={`text-sm font-medium transition-colors ${
                        botStatus.is_running ? 'text-green-600' : 'text-gray-500'
                      }`}>
                        {botStatus.is_running ? 'Running' : 'Stopped'}
                      </span>
                      <div className={`w-2 h-2 rounded-full transition-colors ${
                        botStatus.is_running ? 'bg-green-400 animate-pulse' : 'bg-gray-300'
                      }`}></div>
                    </div>
                    <div className="flex space-x-2">
                      {!botStatus.is_running ? (
                        <button
                          onClick={handleStartBot}
                          disabled={isStartingBot || !hasCurrentAccountToken}
                          className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 transition-colors"
                        >
                          {isStartingBot ? (
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                          ) : (
                            <Play className="w-4 h-4" />
                          )}
                          <span>{isStartingBot ? 'Starting...' : 'Start Bot'}</span>
                        </button>
                      ) : (
                        <button
                          onClick={handleStopBot}
                          disabled={isStoppingBot}
                          className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
                        >
                          {isStoppingBot ? (
                            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                          ) : (
                            <StopCircle className="w-4 h-4" />
                          )}
                          <span>{isStoppingBot ? 'Stopping...' : 'Stop Bot'}</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Bot Status Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-blue-600 font-medium">Strategy</p>
                        <p className="text-lg font-bold text-blue-900">{botStatus.current_strategy}</p>
                        <p className="text-xs text-blue-700">{botStatus.strategy_status}</p>
                      </div>
                      <Target className="w-8 h-8 text-blue-600" />
                    </div>
                  </div>
                  
                  <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border border-green-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-green-600 font-medium">Daily P&L</p>
                        <p className={`text-lg font-bold ${
                          (botStatus.daily_profit - botStatus.daily_loss) > 0 ? 'text-green-900' : 'text-red-600'
                        }`}>
                          ${(botStatus.daily_profit - botStatus.daily_loss).toFixed(2)}
                        </p>
                        <p className="text-xs text-green-700">
                          +${botStatus.daily_profit.toFixed(2)} / -${botStatus.daily_loss.toFixed(2)}
                        </p>
                      </div>
                      <TrendingUp className="w-8 h-8 text-green-600" />
                    </div>
                  </div>
                  
                  <div className="p-4 bg-gradient-to-r from-purple-50 to-violet-50 rounded-lg border border-purple-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-purple-600 font-medium">Active Trades</p>
                        <p className="text-lg font-bold text-purple-900">{botStatus.active_trades_count}</p>
                        <p className="text-xs text-purple-700">
                          Max: {botStatus.settings.max_concurrent_trades}
                        </p>
                      </div>
                      <Activity className="w-8 h-8 text-purple-600" />
                    </div>
                  </div>
                </div>

                {/* Bot Settings Display */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <div className="p-3 bg-gray-50 rounded-lg border">
                    <p className="text-xs text-gray-600 mb-1">Current Stake</p>
                    <p className="font-bold text-gray-900">${botStatus.settings.auto_stake}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg border">
                    <p className="text-xs text-gray-600 mb-1">Win Rate</p>
                    <p className="font-bold text-gray-900">{botStatus.win_rate.toFixed(1)}%</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg border">
                    <p className="text-xs text-gray-600 mb-1">Daily Target</p>
                    <p className="font-bold text-green-600">${botStatus.settings.daily_target}</p>
                  </div>
                  <div className="p-3 bg-gray-50 rounded-lg border">
                    <p className="text-xs text-gray-600 mb-1">Stop Loss</p>
                    <p className="font-bold text-red-600">${botStatus.settings.daily_stop_loss}</p>
                  </div>
                </div>

                {/* Active Trades Section */}
                {activeTrades.length > 0 && (
                  <div className="border-t pt-6">
                    <h4 className="text-md font-semibold text-gray-900 mb-4 flex items-center space-x-2">
                      <Eye className="w-5 h-5 text-blue-600" />
                      <span>Active Trades ({activeTrades.length})</span>
                    </h4>
                    <div className="space-y-3">
                      {activeTrades.map((trade, index) => (
                        <div key={trade.id || index} className="p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border border-yellow-200">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-4">
                                <div>
                                  <p className="font-medium text-gray-900">
                                    {trade.contract_type} - {trade.symbol}
                                  </p>
                                  <p className="text-sm text-gray-600">
                                    Stake: ${trade.stake} | Entry: {trade.entry_price}
                                  </p>
                                </div>
                                <div className="text-right">
                                  <p className="text-sm text-gray-600">Duration</p>
                                  <p className="font-medium">{trade.duration}s</p>
                                </div>
                                <div className="text-right">
                                  <p className="text-sm text-gray-600">Status</p>
                                  <p className={`font-medium ${
                                    trade.status === 'OPEN' ? 'text-blue-600' : 'text-gray-600'
                                  }`}>
                                    {trade.status}
                                  </p>
                                </div>
                              </div>
                            </div>
                            <button
                              onClick={() => handleForceCloseTrade(trade.id)}
                              className="ml-4 px-3 py-1 text-sm bg-red-100 text-red-600 rounded hover:bg-red-200 transition-colors"
                            >
                              Force Close
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
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
                      <button className="w-full h-10 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors font-medium">
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
