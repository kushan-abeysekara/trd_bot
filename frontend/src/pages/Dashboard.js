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
  Pause
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const Dashboard = () => {
  const { user, logout } = useAuth();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [autoTradeEnabled, setAutoTradeEnabled] = useState(user?.auto_trade_enabled || false);
  const [stats] = useState({
    balance: user?.trading_balance || 0,
    profit: 2450.50,
    trades: 127,
    winRate: 78.5
  });

  const handleLogout = () => {
    logout();
  };

  const toggleAutoTrade = () => {
    setAutoTradeEnabled(!autoTradeEnabled);
    // Here you would typically make an API call to update the setting
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
      </nav>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Welcome back, {user?.first_name}!
          </h2>
          <p className="text-gray-600">
            Here's your trading dashboard overview
          </p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Trading Balance</p>
                <p className="text-2xl font-bold text-gray-900">${stats.balance.toFixed(2)}</p>
              </div>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                <DollarSign className="w-6 h-6 text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Profit</p>
                <p className="text-2xl font-bold text-green-600">${stats.profit.toFixed(2)}</p>
              </div>
              <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Total Trades</p>
                <p className="text-2xl font-bold text-gray-900">{stats.trades}</p>
              </div>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-6 h-6 text-purple-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Win Rate</p>
                <p className="text-2xl font-bold text-gray-900">{stats.winRate}%</p>
              </div>
              <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
                <Activity className="w-6 h-6 text-yellow-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Auto Trading Control */}
          <div className="lg:col-span-2">
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
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
              <div className="space-y-3">
                {[1, 2, 3].map((item) => (
                  <div key={item} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <div>
                        <p className="font-medium text-gray-900">Successful Trade</p>
                        <p className="text-sm text-gray-600">EUR/USD - Buy Order</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-green-600">+$45.20</p>
                      <p className="text-sm text-gray-500">2 min ago</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
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
    </div>
  );
};

export default Dashboard;
