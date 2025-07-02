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
  
  // Real-time trading updates
  const [liveStatus, setLiveStatus] = useState(null);
  const [activeTrades, setActiveTrades] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [marketData, setMarketData] = useState(null);
  
  // Contract analysis
  const [contractAnalysis, setContractAnalysis] = useState({});
  const [aiPredictions, setAiPredictions] = useState({});

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

  // Load live trading status with real-time updates
  const loadLiveStatus = useCallback(async () => {
    try {
      const response = await tradingAPI.getLiveTradingStatus();
      if (response.success) {
        setLiveStatus(response.data);
        
        // Update active trades if bot is running
        if (response.data.is_active) {
          await loadActiveTrades();
          await loadMarketData();
          await loadAIPredictions();
        }
      }
    } catch (error) {
      console.error('Error loading live status:', error);
    }
  }, []);

  const loadActiveTrades = async () => {
    try {
      const response = await tradingAPI.getActiveTrades();
      if (response.success) {
        setActiveTrades(response.data);
      }
    } catch (error) {
      console.error('Error loading active trades:', error);
    }
  };

  const loadMarketData = async () => {
    try {
      const response = await tradingAPI.getCurrentMarketData();
      if (response.success) {
        setMarketData(response.data);
      }
    } catch (error) {
      console.error('Error loading market data:', error);
    }
  };

  const loadAIPredictions = async () => {
    try {
      const response = await tradingAPI.getAIPredictions();
      if (response.success) {
        setAiPredictions(response.data);
      }
    } catch (error) {
      console.error('Error loading AI predictions:', error);
    }
  };

  // Start real-time updates when bot is active
  useEffect(() => {
    if (botStatus?.is_active) {
      // Update every 2 seconds when bot is active
      const interval = setInterval(() => {
        loadLiveStatus();
      }, 2000);
      
      updateInterval.current = interval;
      return () => clearInterval(interval);
    } else {
      // Update every 10 seconds when bot is inactive
      const interval = setInterval(() => {
        loadBotStatus();
      }, 10000);
      
      updateInterval.current = interval;
      return () => clearInterval(interval);
    }
  }, [botStatus?.is_active, loadLiveStatus]);

  const startBot = async () => {
    try {
      setIsLoading(true);
      const response = await tradingAPI.startBot();
      
      if (response.success) {
        await loadBotData();
        // Start frequent updates
        loadLiveStatus();
      } else {
        setError(response.message || 'Failed to start bot');
      }
    } catch (error) {
      setError('Failed to start trading bot');
    } finally {
      setIsLoading(false);
    }
  };

  const stopBot = async () => {
    try {
      setIsLoading(true);
      const response = await tradingAPI.stopBot();
      
      if (response.success) {
        await loadBotData();
        setActiveTrades([]);
        setLiveStatus(null);
      } else {
        setError(response.message || 'Failed to stop bot');
      }
    } catch (error) {
      setError('Failed to stop trading bot');
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

  // Render live trading dashboard
  const renderLiveTradingDashboard = () => {
    if (!liveStatus || !liveStatus.is_active) return null;

    return (
      <div className="space-y-6">
        {/* Live Status Bar */}
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-green-800 font-medium">Bot Active - Live Trading</span>
            </div>
            <div className="text-sm text-green-600">
              Last Update: {new Date(liveStatus.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>

        {/* Market Data & AI Predictions */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Current Market Analysis */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Activity className="mr-2" size={20} />
              Market Analysis
            </h3>
            {marketData && (
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Current Price:</span>
                  <span className="font-mono">{marketData.current_price?.toFixed(5)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Volatility:</span>
                  <span className={`font-mono ${marketData.volatility > 0.02 ? 'text-red-600' : 'text-green-600'}`}>
                    {(marketData.volatility * 100).toFixed(3)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Trend:</span>
                  <span className={`font-mono ${marketData.trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {marketData.trend > 0 ? '↗️ Bullish' : '↘️ Bearish'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>RSI:</span>
                  <span className={`font-mono ${marketData.rsi > 70 ? 'text-red-600' : marketData.rsi < 30 ? 'text-green-600' : 'text-gray-600'}`}>
                    {marketData.rsi?.toFixed(1)}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* AI Predictions */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Brain className="mr-2" size={20} />
              AI Predictions
            </h3>
            {aiPredictions && (
              <div className="space-y-3">
                {Object.entries(contractTypes).slice(0, 4).map(([key, contract]) => {
                  const prediction = aiPredictions[key];
                  return (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-sm">{contract.icon} {contract.name}:</span>
                      <div className="text-right">
                        <div className={`text-sm font-medium ${prediction?.direction === 'up' ? 'text-green-600' : 'text-red-600'}`}>
                          {prediction?.direction?.toUpperCase() || 'ANALYZING'}
                        </div>
                        <div className="text-xs text-gray-500">
                          {prediction?.confidence ? `${(prediction.confidence * 100).toFixed(0)}%` : 'N/A'}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        {/* Active Trades */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Eye className="mr-2" size={20} />
            Active Trades ({activeTrades.length})
          </h3>
          {activeTrades.length > 0 ? (
            <div className="space-y-3">
              {activeTrades.map((trade, index) => (
                <div key={index} className="border rounded-lg p-4 bg-gray-50">
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="font-medium">
                        {contractTypes[trade.contract_type]?.icon} {contractTypes[trade.contract_type]?.name}
                      </div>
                      <div className="text-sm text-gray-600">
                        {trade.action?.toUpperCase()} | ${trade.stake_amount?.toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-500">
                        Started: {new Date(trade.start_time).toLocaleTimeString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-sm font-medium ${trade.action === 'call' ? 'text-green-600' : 'text-red-600'}`}>
                        {(trade.confidence * 100).toFixed(0)}% Confidence
                      </div>
                      <div className="text-xs text-gray-500">
                        {trade.contract_id?.slice(-8)}
                      </div>
                    </div>
                  </div>
                  
                  {/* Trade Progress Bar */}
                  <div className="mt-3">
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${Math.min(100, ((Date.now() - new Date(trade.start_time).getTime()) / (trade.duration * 1000)) * 100)}%` 
                        }}
                      ></div>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Duration: {trade.duration}s
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No active trades. Bot is analyzing market conditions...
            </div>
          )}
        </div>

        {/* Strategy Performance */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Strategy Performance</h3>
          {liveStatus.strategy_performance && (
            <div className="grid grid-cols-3 gap-4">
              {Object.entries(liveStatus.strategy_performance).map(([strategy, perf]) => (
                <div key={strategy} className="text-center p-3 bg-gray-50 rounded">
                  <div className="text-sm font-medium">{strategy.replace('_', ' ')}</div>
                  <div className="text-lg font-bold text-green-600">
                    {perf.total_trades > 0 ? `${((perf.wins / perf.total_trades) * 100).toFixed(0)}%` : 'N/A'}
                  </div>
                  <div className="text-xs text-gray-500">
                    {perf.total_trades} trades
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <Bot size={32} className="text-blue-600" />
          <div>
            <h1 className="text-2xl font-bold">AI Trading Bot</h1>
            <p className="text-gray-600">
              {botStatus?.is_active ? 'Live Trading Active' : 'Ready to Trade'}
            </p>
          </div>
        </div>
        
        <div className="flex space-x-3">
          {!botStatus?.is_active ? (
            <button
              onClick={startBot}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              <Play size={20} />
              <span>{isLoading ? 'Starting...' : 'Start Bot'}</span>
            </button>
          ) : (
            <button
              onClick={stopBot}
              disabled={isLoading}
              className="flex items-center space-x-2 bg-red-600 text-white px-6 py-3 rounded-lg hover:bg-red-700 disabled:opacity-50"
            >
              <Square size={20} />
              <span>{isLoading ? 'Stopping...' : 'Stop Bot'}</span>
            </button>
          )}
          
          <button
            onClick={() => setShowSettings(true)}
            className="flex items-center space-x-2 bg-gray-600 text-white px-4 py-3 rounded-lg hover:bg-gray-700"
          >
            <Settings size={20} />
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="text-red-800">{error}</div>
        </div>
      )}

      {/* Live Trading Dashboard */}
      {renderLiveTradingDashboard()}

      {/* Bot Status Overview (when not active) */}
      {!botStatus?.is_active && botStatus && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Bot Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <DollarSign className="mx-auto mb-2 text-blue-600" size={24} />
              <div className="text-lg font-bold">${botStatus.account_balance?.toFixed(2)}</div>
              <div className="text-sm text-gray-600">Account Balance</div>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <Target className="mx-auto mb-2 text-green-600" size={24} />
              <div className="text-lg font-bold">{botStatus.daily_trades || 0}</div>
              <div className="text-sm text-gray-600">Daily Trades</div>
            </div>
            
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <TrendingUp className="mx-auto mb-2 text-purple-600" size={24} />
              <div className={`text-lg font-bold ${(botStatus.daily_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${(botStatus.daily_pnl || 0).toFixed(2)}
              </div>
              <div className="text-sm text-gray-600">Daily P&L</div>
            </div>
          </div>
        </div>
      )}

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
