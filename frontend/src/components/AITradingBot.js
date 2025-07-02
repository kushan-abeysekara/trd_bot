import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Bot, Play, Square, Settings, TrendingUp, TrendingDown, 
  Activity, DollarSign, Target, AlertTriangle,
  Brain, Eye
} from 'lucide-react';
import { tradingAPI } from '../services/api';

const AITradingBot = ({ user }) => {
  // Bot state
  const [botStatus, setBotStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [settings, setSettings] = useState(null);
  
  // Performance data
  const [recentTrades, setRecentTrades] = useState([]);
  const [mlPerformance, setMlPerformance] = useState(null);
  
  // Real-time updates
  const [lastUpdate, setLastUpdate] = useState(null);
  const updateInterval = useRef(null);
  
  // Settings modal
  const [showSettings, setShowSettings] = useState(false);
  const [tempSettings, setTempSettings] = useState({});
  
  // Contract types for display
  const contractTypes = {
    'rise_fall': { name: 'Rise/Fall', color: 'blue', icon: '📈' },
    'touch_no_touch': { name: 'Touch/No Touch', color: 'purple', icon: '🎯' },
    'in_out': { name: 'In/Out', color: 'orange', icon: '🔄' },
    'asians': { name: 'Asian Options', color: 'green', icon: '📊' },
    'digits': { name: 'Digits', color: 'red', icon: '🔢' },
    'reset_call_put': { name: 'Reset Call/Put', color: 'yellow', icon: '🔄' },
    'high_low_ticks': { name: 'High/Low Ticks', color: 'pink', icon: '⚡' },
    'only_ups_downs': { name: 'Only Ups/Downs', color: 'indigo', icon: '↗️' },
    'multipliers': { name: 'Multipliers', color: 'teal', icon: '⚖️' },
    'accumulators': { name: 'Accumulators', color: 'cyan', icon: '💰' }
  };

  // Load initial data
  const loadBotData = useCallback(async () => {
    try {
      setIsLoading(true);
      await Promise.all([
        loadBotStatus(),
        loadBotSettings(),
        loadPerformance(),
        loadRecentTrades(),
        loadMlPerformance()
      ]);
    } catch (error) {
      setError('Failed to load bot data');
      console.error('Error loading bot data:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadBotData();
    
    // Store interval id for cleanup
    const intervalId = updateInterval.current;
    
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [loadBotData]);

  const loadBotStatus = async () => {
    try {
      const response = await tradingAPI.getBotStatus();
      setBotStatus(response.data);
      setLastUpdate(new Date());
    } catch (error) {
      // Don't set error for status updates as bot might not be running
      console.error('Error loading bot status:', error);
    }
  };

  const loadBotSettings = async () => {
    try {
      const response = await tradingAPI.getBotSettings();
      setSettings(response.data);
      setTempSettings(response.data);
    } catch (error) {
      console.error('Error loading bot settings:', error);
    }
  };

  const loadPerformance = async () => {
    try {
      const response = await tradingAPI.getPerformanceAnalytics();
      // Performance data is handled in botStatus
      console.log('Performance data loaded:', response.data);
    } catch (error) {
      console.error('Error loading performance:', error);
    }
  };

  const loadRecentTrades = async () => {
    try {
      const response = await tradingAPI.getTradingHistory({ limit: 10 });
      setRecentTrades(response.data.trades || []);
    } catch (error) {
      console.error('Error loading recent trades:', error);
    }
  };

  const loadMlPerformance = async () => {
    try {
      const response = await tradingAPI.getMlModelPerformance();
      setMlPerformance(response.data);
    } catch (error) {
      console.error('Error loading ML performance:', error);
    }
  };

  const startBot = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await tradingAPI.startTradingBot({
        api_token: user.deriv_api_token,
        account_type: user.deriv_account_type || 'demo',
        settings: tempSettings
      });
      
      if (response.success) {
        await loadBotStatus();
      } else {
        setError(response.message || 'Failed to start bot');
      }
    } catch (error) {
      setError(error.response?.data?.error || 'Failed to start bot');
    } finally {
      setIsLoading(false);
    }
  };

  const stopBot = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await tradingAPI.stopTradingBot();
      
      if (response.success) {
        await loadBotStatus();
      } else {
        setError(response.message || 'Failed to stop bot');
      }
    } catch (error) {
      setError(error.response?.data?.error || 'Failed to stop bot');
    } finally {
      setIsLoading(false);
    }
  };

  const updateBotSettings = async () => {
    try {
      setIsLoading(true);
      const response = await tradingAPI.updateBotSettings(tempSettings);
      
      if (response.success) {
        setSettings(tempSettings);
        setShowSettings(false);
      } else {
        setError('Failed to update settings');
      }
    } catch (error) {
      setError('Failed to update settings');
    } finally {
      setIsLoading(false);
    }
  };

  const testSignal = async () => {
    try {
      setIsLoading(true);
      const response = await tradingAPI.testTradingSignal();
      
      if (response.success) {
        // Show signal test results
        alert(`Signal Test: ${JSON.stringify(response.data, null, 2)}`);
      }
    } catch (error) {
      setError('Failed to test signal');
    } finally {
      setIsLoading(false);
    }
  };

  const retrainModels = async () => {
    try {
      setIsLoading(true);
      const response = await tradingAPI.retrainMlModels();
      
      if (response.success) {
        await loadMlPerformance();
        alert('ML models retrained successfully');
      }
    } catch (error) {
      setError('Failed to retrain models');
    } finally {
      setIsLoading(false);
    }
  };

  const getStrategyIcon = (mode) => {
    const icons = {
      'MA_RSI_TREND': '📈',
      'PRICE_ACTION_BOUNCE': '🔄',
      'RANDOM_ENTRY_SMART_EXIT': '🎲'
    };
    return icons[mode] || '🤖';
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount || 0);
  };

  const formatPercentage = (value) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center transition-colors ${
              botStatus?.is_active ? 'bg-green-100' : 'bg-gray-100'
            }`}>
              <Bot className={`w-6 h-6 transition-colors ${
                botStatus?.is_active ? 'text-green-600' : 'text-gray-400'
              }`} />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">AI Trading Bot</h3>
              <p className="text-sm text-gray-600">
                {botStatus?.is_active ? 'Active • Monitoring markets' : 'Inactive • Ready to trade'}
              </p>
            </div>
          </div>
          
          {/* Status indicator */}
          <div className="flex items-center space-x-4">
            <div className={`w-3 h-3 rounded-full ${
              botStatus?.is_active ? 'bg-green-400 animate-pulse' : 'bg-gray-300'
            }`}></div>
            
            {lastUpdate && (
              <span className="text-xs text-gray-500">
                Updated: {lastUpdate.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="p-4 bg-red-50 border-l-4 border-red-400">
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-400 mr-2" />
            <p className="text-red-700">{error}</p>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-400 hover:text-red-600"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="p-6">
        {/* Control buttons */}
        <div className="flex items-center space-x-4 mb-6">
          {!botStatus?.is_active ? (
            <button
              onClick={startBot}
              disabled={isLoading || !user.deriv_api_token}
              className="flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Play className="w-5 h-5" />
              <span>{isLoading ? 'Starting...' : 'Start Bot'}</span>
            </button>
          ) : (
            <button
              onClick={stopBot}
              disabled={isLoading}
              className="flex items-center space-x-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 transition-colors"
            >
              <Square className="w-5 h-5" />
              <span>{isLoading ? 'Stopping...' : 'Stop Bot'}</span>
            </button>
          )}
          
          <button
            onClick={() => setShowSettings(true)}
            className="flex items-center space-x-2 px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <Settings className="w-5 h-5" />
            <span>Settings</span>
          </button>
          
          <button
            onClick={testSignal}
            disabled={!botStatus?.is_active || isLoading}
            className="flex items-center space-x-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            <Eye className="w-5 h-5" />
            <span>Test Signal</span>
          </button>
          
          <button
            onClick={retrainModels}
            disabled={isLoading}
            className="flex items-center space-x-2 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 transition-colors"
          >
            <Brain className="w-5 h-5" />
            <span>Retrain AI</span>
          </button>
        </div>

        {/* Status cards */}
        {botStatus && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {/* Account Balance */}
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Account Balance</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {formatCurrency(botStatus.account_balance)}
                  </p>
                </div>
                <DollarSign className="w-8 h-8 text-gray-400" />
              </div>
            </div>

            {/* Daily P&L */}
            <div className={`rounded-lg p-4 ${
              botStatus.daily_pnl >= 0 ? 'bg-green-50' : 'bg-red-50'
            }`}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Daily P&L</p>
                  <p className={`text-lg font-semibold ${
                    botStatus.daily_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatCurrency(botStatus.daily_pnl)}
                  </p>
                </div>
                {botStatus.daily_pnl >= 0 ? 
                  <TrendingUp className="w-8 h-8 text-green-400" /> :
                  <TrendingDown className="w-8 h-8 text-red-400" />
                }
              </div>
            </div>

            {/* Win Rate */}
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Win Rate</p>
                  <p className="text-lg font-semibold text-blue-600">
                    {formatPercentage(botStatus.win_rate)}
                  </p>
                </div>
                <Target className="w-8 h-8 text-blue-400" />
              </div>
            </div>

            {/* Total Trades */}
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Daily Trades</p>
                  <p className="text-lg font-semibold text-purple-600">
                    {botStatus.daily_trades}
                  </p>
                </div>
                <Activity className="w-8 h-8 text-purple-400" />
              </div>
            </div>
          </div>
        )}

        {/* Current strategy and streaks */}
        {botStatus && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {/* Current Strategy */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{getStrategyIcon(botStatus.current_mode)}</span>
                <div>
                  <p className="text-sm text-gray-600">Current Strategy</p>
                  <p className="font-semibold text-gray-900">
                    {botStatus.current_mode?.replace('_', ' ') || 'N/A'}
                  </p>
                </div>
              </div>
            </div>

            {/* Win Streak */}
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">🔥</span>
                <div>
                  <p className="text-sm text-gray-600">Win Streak</p>
                  <p className="font-semibold text-green-600">
                    {botStatus.consecutive_wins} wins
                  </p>
                </div>
              </div>
            </div>

            {/* Martingale Step */}
            <div className="bg-orange-50 rounded-lg p-4">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">⚖️</span>
                <div>
                  <p className="text-sm text-gray-600">Martingale Step</p>
                  <p className="font-semibold text-orange-600">
                    Step {botStatus.martingale_step}
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recent trades */}
        {recentTrades.length > 0 && (
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">Recent Trades</h4>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="space-y-3">
                {recentTrades.slice(0, 5).map((trade, index) => (
                  <div key={index} className="flex items-center justify-between py-2 border-b border-gray-200 last:border-b-0">
                    <div className="flex items-center space-x-3">
                      <span className="text-sm">
                        {contractTypes[trade.contract_type]?.icon || '📊'}
                      </span>
                      <div>
                        <p className="text-sm font-medium text-gray-900">
                          {contractTypes[trade.contract_type]?.name || trade.contract_type}
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(trade.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                    
                    <div className="text-right">
                      <p className={`text-sm font-semibold ${
                        trade.success ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {trade.success ? '+' : ''}{formatCurrency(trade.profit_loss)}
                      </p>
                      <p className="text-xs text-gray-500">
                        Stake: {formatCurrency(trade.stake_amount)}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ML Performance */}
        {mlPerformance && Object.keys(mlPerformance).length > 0 && (
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4">ML Model Performance</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(mlPerformance).map(([contractType, perf]) => (
                <div key={contractType} className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-900">
                      {contractTypes[contractType]?.name || contractType}
                    </span>
                    <span>{contractTypes[contractType]?.icon || '📊'}</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Accuracy:</span>
                      <span className="font-medium">{(perf.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Samples:</span>
                      <span className="font-medium">{perf.training_samples}</span>
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-600">Last Trained:</span>
                      <span className="font-medium">
                        {perf.last_trained ? new Date(perf.last_trained).toLocaleDateString() : 'Never'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Settings Modal */}
      {showSettings && settings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Bot Settings</h3>
            
            <div className="space-y-4">
              {/* Daily Stop Loss */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Daily Stop Loss (%)
                </label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  step="0.1"
                  value={tempSettings.daily_stop_loss_percent || 10}
                  onChange={(e) => setTempSettings({
                    ...tempSettings,
                    daily_stop_loss_percent: parseFloat(e.target.value)
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Daily Target */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Daily Target (%)
                </label>
                <input
                  type="number"
                  min="1"
                  max="100"
                  step="0.1"
                  value={tempSettings.daily_target_percent || 20}
                  onChange={(e) => setTempSettings({
                    ...tempSettings,
                    daily_target_percent: parseFloat(e.target.value)
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Base Stake */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Base Stake (%)
                </label>
                <input
                  type="number"
                  min="0.1"
                  max="10"
                  step="0.1"
                  value={tempSettings.base_stake_percent || 2}
                  onChange={(e) => setTempSettings({
                    ...tempSettings,
                    base_stake_percent: parseFloat(e.target.value)
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Enable Martingale */}
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="enable_martingale"
                  checked={tempSettings.enable_martingale || false}
                  onChange={(e) => setTempSettings({
                    ...tempSettings,
                    enable_martingale: e.target.checked
                  })}
                  className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="enable_martingale" className="ml-2 text-sm font-medium text-gray-700">
                  Enable Smart Martingale
                </label>
              </div>

              {/* Martingale Multiplier */}
              {tempSettings.enable_martingale && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Martingale Multiplier
                  </label>
                  <input
                    type="number"
                    min="1.1"
                    max="3"
                    step="0.1"
                    value={tempSettings.martingale_multiplier || 1.5}
                    onChange={(e) => setTempSettings({
                      ...tempSettings,
                      martingale_multiplier: parseFloat(e.target.value)
                    })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              )}

              {/* Cooldown */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cooldown After Losses
                </label>
                <input
                  type="number"
                  min="0"
                  max="10"
                  value={tempSettings.cool_down_after_loss || 3}
                  onChange={(e) => setTempSettings({
                    ...tempSettings,
                    cool_down_after_loss: parseInt(e.target.value)
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>

            {/* Modal buttons */}
            <div className="flex items-center space-x-4 mt-6">
              <button
                onClick={updateBotSettings}
                disabled={isLoading}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {isLoading ? 'Saving...' : 'Save Settings'}
              </button>
              <button
                onClick={() => setShowSettings(false)}
                className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AITradingBot;
