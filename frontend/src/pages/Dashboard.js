import React, { useState, useEffect } from 'react';
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
  AlertCircle
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { tradingAPI } from '../services/api';
import ApiTokenSetup from '../components/ApiTokenSetup';
import VolatilityChart from '../components/VolatilityChart';
import LastDigitDisplay from '../components/LastDigitDisplay';
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

  // Check API token availability for both account types
  const hasDemoToken = user?.has_demo_token || false;
  const hasRealToken = user?.has_real_token || false;
  const hasCurrentAccountToken = currentAccountType === 'demo' ? hasDemoToken : hasRealToken;

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
  }, [hasCurrentAccountToken, currentAccountType]);

  const fetchBalance = async () => {
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
  };

  const fetchStats = async () => {
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
  };

  const fetchRecentActivity = async () => {
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
  };

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
  }, [hasApiToken]);

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

  const handleLastDigitUpdate = (digit, indexName, price) => {
    setLastDigit(digit);
    setCurrentIndexName(indexName);
    setCurrentPrice(price);
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

            {/* Last Digit Display - Center */}
            <div className="hidden md:flex items-center space-x-4">
              {currentPrice && (
                <div className="flex items-center space-x-3 px-3 py-1 bg-blue-50 rounded-lg">
                  <div className="text-right">
                    <p className="text-xs text-blue-600 font-medium">Current Price</p>
                    <p className="text-sm font-bold text-blue-900">{currentPrice.toFixed(2)}</p>
                  </div>
                  <div className="h-8 w-px bg-blue-200"></div>
                </div>
              )}
              <LastDigitDisplay lastDigit={lastDigit} indexName={currentIndexName} />
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
        <div className="md:hidden mb-6 space-y-3">
          {currentPrice && (
            <div className="flex justify-center">
              <div className="flex items-center space-x-4 px-4 py-2 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="text-center">
                  <p className="text-xs text-blue-600 font-medium">Current Price</p>
                  <p className="text-lg font-bold text-blue-900">{currentPrice.toFixed(2)}</p>
                </div>
                <div className="h-10 w-px bg-blue-200"></div>
                <div className="text-center">
                  <p className="text-xs text-blue-600 font-medium">2nd Decimal</p>
                  <p className="text-lg font-bold text-blue-900">{lastDigit ?? '-'}</p>
                </div>
              </div>
            </div>
          )}
          <div className="flex justify-center">
            <LastDigitDisplay lastDigit={lastDigit} indexName={currentIndexName} />
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {/* Balance Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <div>
                <p className="text-sm font-medium text-gray-600">
                  {accountBalance ? 
                    `${accountBalance.account_type.charAt(0).toUpperCase() + accountBalance.account_type.slice(1)} Balance` : 
                    `${currentAccountType.charAt(0).toUpperCase() + currentAccountType.slice(1)} Balance`
                  }
                </p>
                {accountBalance ? (
                  <div>
                    <p className="text-2xl font-bold text-blue-600">
                      {accountBalance.currency} {accountBalance.balance.toFixed(2)}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Account: {accountBalance.account_id}
                    </p>
                  </div>
                ) : hasCurrentAccountToken ? (
                  <p className="text-2xl font-bold text-gray-400">
                    {isLoadingBalance ? 'Loading...' : 'N/A'}
                  </p>
                ) : (
                  <p className="text-2xl font-bold text-gray-400">
                    No API Token
                  </p>
                )}
              </div>
              <div className="flex flex-col items-end space-y-1">
                <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                  accountBalance ? 'bg-green-100' : 'bg-gray-100'
                }`}>
                  <DollarSign className={`w-6 h-6 ${
                    accountBalance ? 'text-green-600' : 'text-gray-400'
                  }`} />
                </div>
                {hasCurrentAccountToken && (
                  <button
                    onClick={fetchBalance}
                    disabled={isLoadingBalance}
                    className="text-xs text-blue-600 hover:text-blue-700 disabled:opacity-50 flex items-center"
                    title="Refresh balance"
                  >
                    <RefreshCw className={`w-3 h-3 ${isLoadingBalance ? 'animate-spin' : ''}`} />
                  </button>
                )}
              </div>
            </div>
            {balanceError && (
              <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-600">
                {balanceError}
              </div>
            )}
            {accountBalance && (
              <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                <span>Last updated: {new Date(accountBalance.last_updated).toLocaleTimeString()}</span>
                {accountBalance.account_type === 'demo' && (
                  <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded">Demo</span>
                )}
                {accountBalance.account_type === 'real' && (
                  <span className="bg-green-100 text-green-800 px-2 py-1 rounded">Real</span>
                )}
              </div>
            )}
            {isLoadingBalance && (
              <div className="mt-2 flex items-center text-xs text-blue-600">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600 mr-2"></div>
                Updating balance...
              </div>
            )}
          </div>

          {/* Total Profit Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Profit</p>
                <p className={`text-2xl font-bold ${
                  stats.profit > 0 ? 'text-green-600' : stats.profit < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  {hasCurrentAccountToken ? 
                    (isLoadingStats ? 'Loading...' : `$${stats.profit.toFixed(2)}`) : 
                    'No Data'
                  }
                </p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          {/* Total Trades Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Trades</p>
                <p className="text-2xl font-bold text-gray-900">
                  {hasCurrentAccountToken ? 
                    (isLoadingStats ? 'Loading...' : stats.trades) : 
                    'No Data'
                  }
                </p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>

          {/* Win Rate Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Win Rate</p>
                <p className="text-2xl font-bold text-gray-900">
                  {hasCurrentAccountToken ? 
                    (isLoadingStats ? 'Loading...' : `${stats.winRate.toFixed(1)}%`) : 
                    'No Data'
                  }
                </p>
              </div>
              <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-yellow-600" />
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
            
            {/* Auto Trading Control */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-900">Auto Trading</h3>
                <div className="flex items-center space-x-2">
                  <span className={`text-sm font-medium ${autoTradeEnabled ? 'text-green-600' : 'text-gray-500'}`}>
                    {autoTradeEnabled ? 'Active' : 'Inactive'}
                  </span>
                  <button
                    onClick={toggleAutoTrade}
                    className={`p-2 rounded-lg transition-colors ${
                      autoTradeEnabled 
                        ? 'bg-green-100 text-green-600 hover:bg-green-200' 
                        : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                    }`}
                  >
                    {autoTradeEnabled ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* Trading Status */}
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Bot className="w-8 h-8 text-blue-600" />
                      <div>
                        <p className="font-medium text-gray-900">AI Trading Bot</p>
                        <p className="text-sm text-gray-600">
                          {autoTradeEnabled ? 'Monitoring markets and executing trades' : 'Waiting for activation'}
                        </p>
                      </div>
                    </div>
                    <div className={`w-3 h-3 rounded-full ${autoTradeEnabled ? 'bg-green-400 animate-pulse' : 'bg-gray-300'}`}></div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-3 border border-gray-200 rounded-lg">
                    <p className="text-sm text-gray-600">Risk Level</p>
                    <p className="font-semibold text-yellow-600">Medium</p>
                  </div>
                  <div className="p-3 border border-gray-200 rounded-lg">
                    <p className="text-sm text-gray-600">Strategy</p>
                    <p className="font-semibold text-blue-600">AI Adaptive</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white rounded-lg shadow p-6 mt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
                <button
                  onClick={fetchRecentActivity}
                  disabled={isLoadingActivity || !hasCurrentAccountToken}
                  className="text-sm text-blue-600 hover:text-blue-700 disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoadingActivity ? 'animate-spin' : ''}`} />
                </button>
              </div>
              
              {!hasCurrentAccountToken ? (
                <div className="text-center py-8 text-gray-500">
                  <Activity className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>Configure API token to view recent activity</p>
                </div>
              ) : isLoadingActivity ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                  <p className="text-gray-500 mt-2">Loading activities...</p>
                </div>
              ) : recentActivity.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <Activity className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>No recent activities</p>
                  <p className="text-sm">Your trading activities will appear here</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {recentActivity.map((activity) => (
                    <div key={activity.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div className={`w-2 h-2 rounded-full ${
                          activity.type === 'profit' ? 'bg-green-400' : 
                          activity.type === 'loss' ? 'bg-red-400' : 'bg-blue-400'
                        }`}></div>
                        <div>
                          <p className="font-medium text-gray-900">{activity.title}</p>
                          <p className="text-sm text-gray-600">{activity.description}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className={`font-medium ${
                          activity.amount > 0 ? 'text-green-600' : 
                          activity.amount < 0 ? 'text-red-600' : 'text-gray-600'
                        }`}>
                          {activity.amount > 0 ? '+' : ''}${activity.amount.toFixed(2)}
                        </p>
                        <p className="text-sm text-gray-500">{activity.time}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* API Configuration */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">API Configuration</h3>
              {hasApiToken ? (
                <div className="space-y-4">
                  {/* Overall Status */}
                  <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Shield className="w-5 h-5 text-green-600" />
                        <span className="font-medium text-green-800">API Configured</span>
                      </div>
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    </div>
                    <p className="text-sm text-green-700 mt-1">
                      Your Deriv API token is active and ready for trading
                    </p>
                  </div>

                  {/* API Status Details */}
                  <div className="space-y-3">
                    {/* Demo Account Status */}
                    <div className="flex items-center justify-between p-2 border border-gray-200 rounded">
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-600">Demo API</span>
                        {user?.deriv_account_type === 'demo' || accountBalance?.account_type === 'demo' ? (
                          <div className="flex items-center space-x-1">
                            <Shield className="w-4 h-4 text-green-600" />
                            <span className="text-xs text-green-600 font-medium">Active</span>
                          </div>
                        ) : (
                          <span className="text-xs text-gray-400">Not Set</span>
                        )}
                      </div>
                      {user?.deriv_account_type === 'demo' && (
                        <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded">Current</span>
                      )}
                    </div>

                    {/* Real Account Status */}
                    <div className="flex items-center justify-between p-2 border border-gray-200 rounded">
                      <div className="flex items-center space-x-2">
                        <span className="text-gray-600">Real API</span>
                        {user?.deriv_account_type === 'real' || accountBalance?.account_type === 'real' ? (
                          <div className="flex items-center space-x-1">
                            <Shield className="w-4 h-4 text-green-600" />
                            <span className="text-xs text-green-600 font-medium">Active</span>
                          </div>
                        ) : (
                          <span className="text-xs text-gray-400">Not Set</span>
                        )}
                      </div>
                      {user?.deriv_account_type === 'real' && (
                        <span className="text-xs text-red-600 bg-red-100 px-2 py-1 rounded">Current</span>
                      )}
                    </div>

                    {accountBalance?.account_id && (
                      <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span className="text-gray-600">Account ID</span>
                        <span className="text-sm text-gray-900 font-mono">{accountBalance.account_id}</span>
                      </div>
                    )}
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setShowApiSetup(true)}
                      className="flex-1 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded px-3 py-2 transition-colors"
                    >
                      Manage APIs
                    </button>
                    <button
                      onClick={handleRemoveApiToken}
                      className="flex-1 text-red-600 hover:text-red-700 text-sm font-medium border border-red-200 hover:border-red-300 rounded px-3 py-2 transition-colors"
                    >
                      Remove API
                    </button>
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                    <Key className="w-6 h-6 text-green-600" />
                  </div>
                  <div className="p-3 bg-green-50 border border-green-200 rounded-lg mb-3">
                    <p className="text-green-700 font-medium mb-2">Ready to Configure</p>
                    <p className="text-sm text-green-600">
                      Setup your Deriv API tokens to enable automated trading
                    </p>
                  </div>
                  <button
                    onClick={() => setShowApiSetup(true)}
                    className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors font-medium"
                  >
                    Setup API Tokens
                  </button>
                  <p className="text-xs text-gray-500 mt-2">
                    You can configure both Demo and Real API tokens
                  </p>
                </div>
              )}
            </div>

            {/* Account Status */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Status</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Verification</span>
                  <div className="flex items-center space-x-2">
                    <Shield className="w-4 h-4 text-green-600" />
                    <span className="text-green-600">Verified</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Plan</span>
                  <span className="text-blue-600 font-medium">Premium</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">API Status</span>
                  <span className="text-green-600">Connected</span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
              <div className="space-y-2">
                <button className="w-full text-left p-3 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors">
                  <div className="flex items-center space-x-3">
                    <DollarSign className="w-4 h-4 text-blue-600" />
                    <span>Add Funds</span>
                  </div>
                </button>
                <button className="w-full text-left p-3 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors">
                  <div className="flex items-center space-x-3">
                    <BarChart3 className="w-4 h-4 text-green-600" />
                    <span>View Reports</span>
                  </div>
                </button>
                <button className="w-full text-left p-3 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors">
                  <div className="flex items-center space-x-3">
                    <Settings className="w-4 h-4 text-gray-600" />
                    <span>Trading Settings</span>
                  </div>
                </button>
                <button className="w-full text-left p-3 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors">
                  <div className="flex items-center space-x-3">
                    <Bell className="w-4 h-4 text-purple-600" />
                    <span>Notifications</span>
                  </div>
                </button>
              </div>
            </div>

            {/* Deriv Connection Status */}
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Deriv Integration</h3>
              <div className="text-center">
                {user?.deriv_account_id ? (
                  <div>
                    <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-3">
                      <Shield className="w-6 h-6 text-green-600" />
                    </div>
                    <p className="text-green-600 font-medium">Connected</p>
                    <p className="text-sm text-gray-600 mt-1">Account: {user.deriv_account_id}</p>
                  </div>
                ) : (
                  <div>
                    <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-3">
                      <Shield className="w-6 h-6 text-orange-600" />
                    </div>
                    <p className="text-orange-600 font-medium">Not Connected</p>
                    <button className="mt-2 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors">
                      Connect Deriv Account
                    </button>
                  </div>
                )}
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
