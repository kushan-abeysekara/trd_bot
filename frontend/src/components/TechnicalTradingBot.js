import React, { useState, useEffect, useCallback } from 'react';
import { toast } from 'react-hot-toast';
import { Play, Square, Activity, BarChart3, Settings, RefreshCw } from 'lucide-react';
import { tradingAPI } from '../services/api';

const TechnicalTradingBot = ({ user }) => {
  const [botStatus, setBotStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [tradingHistory, setTradingHistory] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [settings, setSettings] = useState({
    base_stake: 1.0,
    daily_stop_loss: 100.0,
    daily_target: 200.0,
    risk_per_trade: 2.0,
    account_type: 'demo'
  });
  const [performance, setPerformance] = useState({
    total_profit: 0,
    total_trades: 0,
    win_rate: 0,
    today_profit: 0,
    today_trades: 0
  });

  // Available strategies
  const strategies = [
    { value: 'adaptive_mean_reversion', name: 'Adaptive Mean Reversion', icon: '🔄' },
    { value: 'rsi_momentum_breakout', name: 'RSI Momentum Breakout', icon: '📈' },
    { value: 'bollinger_band_squeeze', name: 'Bollinger Band Squeeze', icon: '🎯' },
    { value: 'macd_histogram_divergence', name: 'MACD Histogram Divergence', icon: '📊' },
    { value: 'volatility_expansion_scalp', name: 'Volatility Expansion Scalp', icon: '⚡' },
    { value: 'tick_velocity_momentum', name: 'Tick Velocity Momentum', icon: '🏃' },
    { value: 'support_resistance_bounce', name: 'Support Resistance Bounce', icon: '🏀' },
    { value: 'ema_crossover_micro', name: 'EMA Crossover Micro', icon: '✂️' },
    { value: 'williams_r_extreme', name: 'Williams R Extreme', icon: '🎪' },
    { value: 'stochastic_divergence', name: 'Stochastic Divergence', icon: '🌊' },
    { value: 'volume_price_trend', name: 'Volume Price Trend', icon: '📈' },
    { value: 'microtrend_reversal', name: 'Microtrend Reversal', icon: '🔀' },
    { value: 'high_frequency_scalp', name: 'High Frequency Scalp', icon: '⚡' },
    { value: 'neural_pattern_recognition', name: 'Neural Pattern Recognition', icon: '🧠' },
    { value: 'adaptive_multi_timeframe', name: 'Adaptive Multi-Timeframe', icon: '⏰' }
  ];

  // Strategy descriptions
  const strategyDescriptions = {
    'adaptive_mean_reversion': 'Uses adaptive thresholds for mean reversion trading',
    'rsi_momentum_breakout': 'Combines RSI signals with momentum indicators',
    'bollinger_band_squeeze': 'Detects low volatility periods before breakouts',
    'macd_histogram_divergence': 'Uses MACD histogram divergences for entries',
    'volatility_expansion_scalp': 'Scalps during volatility expansion phases',
    'tick_velocity_momentum': 'Uses tick velocity for momentum detection',
    'support_resistance_bounce': 'Trades bounces off key support/resistance levels',
    'ema_crossover_micro': 'Micro-timeframe EMA crossover strategy',
    'williams_r_extreme': 'Uses Williams %R extreme readings',
    'stochastic_divergence': 'Detects divergences between price and stochastic oscillator',
    'volume_price_trend': 'Uses volume-price relationships for signal confirmation',
    'microtrend_reversal': 'Captures very short-term trend reversals',
    'high_frequency_scalp': 'Exploits very short-term price inefficiencies',
    'neural_pattern_recognition': 'AI-based pattern recognition for complex signals',
    'adaptive_multi_timeframe': 'Combines signals from multiple timeframes'
  };

  // Load bot status
  const loadBotStatus = useCallback(async () => {
    try {
      const response = await tradingAPI.getTechnicalBotStatus();
      if (response.success) {
        setBotStatus(response.data);
      }
    } catch (error) {
      console.error('Error loading bot status:', error);
    }
  }, []);

  // Load trading history
  const loadTradingHistory = useCallback(async () => {
    try {
      const response = await tradingAPI.getTechnicalTradingHistory();
      if (response.success) {
        setTradingHistory(response.data);
      }
    } catch (error) {
      console.error('Error loading trading history:', error);
    }
  }, []);

  // Load performance data
  const loadPerformance = useCallback(async () => {
    try {
      const response = await tradingAPI.getTechnicalBotPerformance();
      if (response.success) {
        setPerformance(response.data);
      }
    } catch (error) {
      console.error('Error loading performance:', error);
    }
  }, []);

  // Initialize component
  useEffect(() => {
    loadBotStatus();
    loadTradingHistory();
    loadPerformance();

    // Set up polling for real-time updates
    const interval = setInterval(() => {
      loadBotStatus();
      loadPerformance();
    }, 5000);

    return () => clearInterval(interval);
  }, [loadBotStatus, loadTradingHistory, loadPerformance]);

  // Start bot
  const handleStartBot = async () => {
    if (!user?.deriv_api_token) {
      toast.error('Please set up your Deriv API token first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await tradingAPI.startTechnicalBot(settings);
      if (response.success) {
        toast.success('Technical Trading Bot started successfully!');
        await loadBotStatus();
      } else {
        toast.error(response.message || 'Failed to start bot');
      }
    } catch (error) {
      toast.error('Error starting bot: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Stop bot
  const handleStopBot = async () => {
    setIsLoading(true);
    try {
      const response = await tradingAPI.stopTechnicalBot();
      if (response.success) {
        toast.success('Technical Trading Bot stopped');
        await loadBotStatus();
      } else {
        toast.error(response.message || 'Failed to stop bot');
      }
    } catch (error) {
      toast.error('Error stopping bot: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Update settings
  const handleSettingsUpdate = async (newSettings) => {
    try {
      const response = await tradingAPI.updateTechnicalBotSettings(newSettings);
      if (response.success) {
        setSettings(newSettings);
        toast.success('Settings updated successfully');
        setShowSettings(false);
      } else {
        toast.error(response.message || 'Failed to update settings');
      }
    } catch (error) {
      toast.error('Error updating settings: ' + error.message);
    }
  };

  // Change strategy
  const handleStrategyChange = async (strategy) => {
    try {
      const response = await tradingAPI.changeTechnicalBotStrategy(strategy);
      if (response.success) {
        toast.success(`Strategy changed to ${strategies.find(s => s.value === strategy)?.name}`);
        await loadBotStatus();
      } else {
        toast.error(response.message || 'Failed to change strategy');
      }
    } catch (error) {
      toast.error('Error changing strategy: ' + error.message);
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(amount);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center space-x-3">
          <Activity className="h-6 w-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-800">Technical Trading Bot</h2>
          {botStatus?.is_running && (
            <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
              ACTIVE
            </span>
          )}
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors flex items-center space-x-1"
          >
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </button>
          <button
            onClick={() => {
              setShowHistory(!showHistory);
              if (!showHistory) loadTradingHistory();
            }}
            className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors flex items-center space-x-1"
          >
            <BarChart3 className="h-4 w-4" />
            <span>History</span>
          </button>
          <button
            onClick={() => {
              loadBotStatus();
              loadPerformance();
              loadTradingHistory();
            }}
            className="px-3 py-1 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Performance Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg">
          <div className="text-sm text-green-600 font-medium">Total Profit</div>
          <div className={`text-lg font-bold ${performance.total_profit >= 0 ? 'text-green-700' : 'text-red-700'}`}>
            {formatCurrency(performance.total_profit)}
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg">
          <div className="text-sm text-blue-600 font-medium">Total Trades</div>
          <div className="text-lg font-bold text-blue-700">{performance.total_trades}</div>
        </div>
        <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg">
          <div className="text-sm text-purple-600 font-medium">Win Rate</div>
          <div className="text-lg font-bold text-purple-700">{(performance.win_rate * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gradient-to-r from-yellow-50 to-yellow-100 p-4 rounded-lg">
          <div className="text-sm text-yellow-600 font-medium">Today's Profit</div>
          <div className={`text-lg font-bold ${performance.today_profit >= 0 ? 'text-green-700' : 'text-red-700'}`}>
            {formatCurrency(performance.today_profit)}
          </div>
        </div>
        <div className="bg-gradient-to-r from-indigo-50 to-indigo-100 p-4 rounded-lg">
          <div className="text-sm text-indigo-600 font-medium">Today's Trades</div>
          <div className="text-lg font-bold text-indigo-700">{performance.today_trades}</div>
        </div>
      </div>

      {/* Bot Control */}
      <div className="flex items-center justify-between bg-gray-50 p-4 rounded-lg mb-6">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Status:</span>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              botStatus?.is_running
                ? 'bg-green-100 text-green-800'
                : 'bg-red-100 text-red-800'
            }`}>
              {botStatus?.is_running ? 'Running' : 'Stopped'}
            </span>
          </div>
          {botStatus?.current_strategy && (
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium text-gray-700">Strategy:</span>
              <span className="text-sm text-blue-600 font-medium">
                {strategies.find(s => s.value === botStatus.current_strategy)?.name || botStatus.current_strategy}
              </span>
            </div>
          )}
        </div>
        <div className="flex space-x-2">
          {!botStatus?.is_running ? (
            <button
              onClick={handleStartBot}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              <Play className="h-4 w-4" />
              <span>{isLoading ? 'Starting...' : 'Start Bot'}</span>
            </button>
          ) : (
            <button
              onClick={handleStopBot}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors disabled:opacity-50"
            >
              <Square className="h-4 w-4" />
              <span>{isLoading ? 'Stopping...' : 'Stop Bot'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Current Trade Info */}
      {botStatus?.current_trade && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
          <h3 className="text-sm font-medium text-blue-800 mb-2">Current Trade</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-blue-600">Symbol:</span>
              <span className="ml-2 font-medium">{botStatus.current_trade.symbol}</span>
            </div>
            <div>
              <span className="text-blue-600">Direction:</span>
              <span className={`ml-2 font-medium ${
                botStatus.current_trade.direction === 'CALL' ? 'text-green-600' : 'text-red-600'
              }`}>
                {botStatus.current_trade.direction}
              </span>
            </div>
            <div>
              <span className="text-blue-600">Stake:</span>
              <span className="ml-2 font-medium">{formatCurrency(botStatus.current_trade.stake)}</span>
            </div>
            <div>
              <span className="text-blue-600">Strategy:</span>
              <span className="ml-2 font-medium">{botStatus.current_trade.strategy}</span>
            </div>
          </div>
        </div>
      )}

      {/* Settings Panel */}
      {showSettings && (
        <div className="bg-gray-50 border rounded-lg p-4 mb-6">
          <h3 className="text-lg font-medium text-gray-800 mb-4">Bot Settings</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base Stake ($)
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                value={settings.base_stake}
                onChange={(e) => setSettings({...settings, base_stake: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Daily Stop Loss ($)
              </label>
              <input
                type="number"
                step="1"
                min="1"
                value={settings.daily_stop_loss}
                onChange={(e) => setSettings({...settings, daily_stop_loss: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Daily Target ($)
              </label>
              <input
                type="number"
                step="1"
                min="1"
                value={settings.daily_target}
                onChange={(e) => setSettings({...settings, daily_target: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Risk Per Trade (%)
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="10"
                value={settings.risk_per_trade}
                onChange={(e) => setSettings({...settings, risk_per_trade: parseFloat(e.target.value)})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Account Type
              </label>
              <select
                value={settings.account_type}
                onChange={(e) => setSettings({...settings, account_type: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="demo">Demo Account</option>
                <option value="real">Real Account</option>
              </select>
            </div>
          </div>
          <div className="flex space-x-2 mt-4">
            <button
              onClick={() => handleSettingsUpdate(settings)}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
            >
              Save Settings
            </button>
            <button
              onClick={() => setShowSettings(false)}
              className="bg-gray-300 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-400 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Strategy Selection */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-800 mb-3">Trading Strategies</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {strategies.map((strategy) => (
            <div
              key={strategy.value}
              className={`p-3 border rounded-lg cursor-pointer transition-all ${
                botStatus?.current_strategy === strategy.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
              }`}
              onClick={() => handleStrategyChange(strategy.value)}
            >
              <div className="flex items-center space-x-2 mb-1">
                <span className="text-lg">{strategy.icon}</span>
                <span className="font-medium text-gray-800">{strategy.name}</span>
              </div>
              <p className="text-xs text-gray-600">
                {strategyDescriptions[strategy.value]}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Trading History */}
      {showHistory && (
        <div className="border-t pt-6">
          <h3 className="text-lg font-medium text-gray-800 mb-4">Trading History</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-gray-50">
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">Time</th>
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">Symbol</th>
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">Direction</th>
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">Stake</th>
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">Strategy</th>
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">Result</th>
                  <th className="border p-2 text-left text-sm font-medium text-gray-700">P&L</th>
                </tr>
              </thead>
              <tbody>
                {tradingHistory.map((trade, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="border p-2 text-sm">{formatDate(trade.timestamp)}</td>
                    <td className="border p-2 text-sm font-medium">{trade.symbol}</td>
                    <td className="border p-2 text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        trade.direction === 'CALL'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {trade.direction}
                      </span>
                    </td>
                    <td className="border p-2 text-sm">{formatCurrency(trade.stake)}</td>
                    <td className="border p-2 text-sm">{trade.strategy}</td>
                    <td className="border p-2 text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        trade.result === 'WIN'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {trade.result}
                      </span>
                    </td>
                    <td className="border p-2 text-sm">
                      <span className={`font-medium ${
                        trade.profit >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatCurrency(trade.profit)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {tradingHistory.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No trades yet. Start the bot to begin trading.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default TechnicalTradingBot;
