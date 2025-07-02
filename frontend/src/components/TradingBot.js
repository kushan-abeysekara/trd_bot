import React, { useState, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import { tradingAPI } from '../services/api';

const TradingBot = ({ user }) => {
  // Bot state
  const [botStatus, setBotStatus] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [botSettings, setBotSettings] = useState({
    daily_stop_loss_percent: 10.0,
    daily_target_percent: 20.0,
    base_stake_percent: 2.0,
    max_stake_percent: 10.0,
    cool_down_after_loss: 3,
    strategy_switch_wins: 3,
    strategy_switch_losses: 2,
    reevaluate_trades: 10,
    enable_martingale: true,
    martingale_multiplier: 1.5,
    max_martingale_steps: 3
  });
  
  // UI state
  const [showSettings, setShowSettings] = useState(false);
  const [showPerformance, setShowPerformance] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  
  // Performance data
  const [performanceData, setPerformanceData] = useState(null);
  const [tradingHistory, setTradingHistory] = useState([]);
  const [mlModelPerformance, setMlModelPerformance] = useState(null);
  
  // Real-time updates
  const intervalRef = useRef(null);
  
  useEffect(() => {
    fetchBotStatus();
    fetchBotSettings();
    
    // Set up real-time updates
    intervalRef.current = setInterval(() => {
      if (botStatus?.is_active) {
        fetchBotStatus();
      }
    }, 5000); // Update every 5 seconds
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [botStatus?.is_active]);
  
  const fetchBotStatus = async () => {
    try {
      const response = await tradingAPI.getBotStatus();
      if (response.data && response.data.success) {
        setBotStatus(response.data.data);
      } else {
        setBotStatus({
          is_active: false,
          current_mode: null,
          account_balance: 0,
          daily_pnl: 0,
          daily_trades: 0,
          win_rate: 0,
          consecutive_wins: 0,
          consecutive_losses: 0
        });
      }
    } catch (error) {
      console.error('Failed to fetch bot status:', error);
      setBotStatus({
        is_active: false,
        current_mode: null,
        account_balance: 0,
        daily_pnl: 0,
        daily_trades: 0,
        win_rate: 0,
        consecutive_wins: 0,
        consecutive_losses: 0
      });
    }
  };
  
  const fetchBotSettings = async () => {
    try {
      const response = await tradingAPI.getBotSettings();
      if (response.data && response.data.success) {
        setBotSettings(response.data.data);
      }
    } catch (error) {
      console.error('Failed to fetch bot settings:', error);
    }
  };
  
  const fetchPerformanceData = async () => {
    try {
      const response = await tradingAPI.getPerformanceAnalytics();
      if (response.data && response.data.success) {
        setPerformanceData(response.data.data);
      }
    } catch (error) {
      console.error('Failed to fetch performance data:', error);
      toast.error('Failed to load performance data');
    }
  };
  
  const fetchTradingHistory = async () => {
    try {
      const response = await tradingAPI.getTradingHistory();
      if (response.data && response.data.success) {
        setTradingHistory(response.data.trades || response.data.data || []);
      }
    } catch (error) {
      console.error('Failed to fetch trading history:', error);
      toast.error('Failed to load trading history');
    }
  };
  
  const fetchMlModelPerformance = async () => {
    try {
      const response = await tradingAPI.getMlModelPerformance();
      if (response.data && response.data.success) {
        setMlModelPerformance(response.data.data);
      }
    } catch (error) {
      console.error('Failed to fetch ML model performance:', error);
    }
  };
  
  const startBot = async () => {
    setIsLoading(true);
    try {
      const response = await tradingAPI.startTradingBot({
        account_type: user?.deriv_account_type || 'demo',
        settings: botSettings
      });
      
      if (response.data && response.data.success) {
        toast.success('Trading bot started successfully!');
        fetchBotStatus();
      } else {
        toast.error(response.data?.error || 'Failed to start trading bot');
      }
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to start trading bot');
    } finally {
      setIsLoading(false);
    }
  };
  
  const stopBot = async () => {
    setIsLoading(true);
    try {
      const response = await tradingAPI.stopTradingBot();
      
      if (response.data && response.data.success) {
        toast.success('Trading bot stopped successfully!');
        fetchBotStatus();
      } else {
        toast.error(response.data?.error || 'Failed to stop trading bot');
      }
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to stop trading bot');
    } finally {
      setIsLoading(false);
    }
  };
  
  const updateSettings = async () => {
    try {
      const response = await tradingAPI.updateBotSettings(botSettings);
      
      if (response.data && response.data.success) {
        toast.success('Bot settings updated successfully!');
        setShowSettings(false);
      } else {
        toast.error(response.data?.error || 'Failed to update settings');
      }
    } catch (error) {
      toast.error(error.response?.data?.error || 'Failed to update settings');
    }
  };
  
  const retrainModels = async () => {
    setIsLoading(true);
    try {
      const response = await tradingAPI.retrainMlModels();
      
      if (response.data && response.data.success) {
        toast.success('ML models retrained successfully!');
        // Fetch updated model performance after retraining
        await fetchMlModelPerformance();
        await fetchBotStatus(); // Update bot status to reflect new models
      } else {
        toast.error(response.data?.error || 'Failed to retrain models');
      }
    } catch (error) {
      console.error('Retrain error:', error);
      toast.error(error.response?.data?.error || 'Failed to retrain models');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSettingChange = (key, value) => {
    setBotSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };
  
  // Strategy mapping and colors
  const strategyMapping = {
    'MA_RSI_TREND': {
      name: 'MA + RSI Trend',
      description: 'Moving Average & RSI based trending strategy',
      color: 'bg-blue-500',
      textColor: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
      icon: '📈'
    },
    'PRICE_ACTION_BOUNCE': {
      name: 'Price Action Bounce',
      description: 'Support/resistance bounce strategy',
      color: 'bg-purple-500',
      textColor: 'text-purple-600',
      bgColor: 'bg-purple-50',
      borderColor: 'border-purple-200',
      icon: '🔄'
    },
    'RANDOM_ENTRY_SMART_EXIT': {
      name: 'Smart Exit Strategy',
      description: 'Random entry with intelligent exit',
      color: 'bg-orange-500',
      textColor: 'text-orange-600',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      icon: '🎯'
    }
  };

  const getStrategyInfo = (mode) => {
    return strategyMapping[mode] || {
      name: 'Unknown Strategy',
      description: 'Strategy not recognized',
      color: 'bg-gray-500',
      textColor: 'text-gray-600',
      bgColor: 'bg-gray-50',
      borderColor: 'border-gray-200',
      icon: '❓'
    };
  };

  const getModeColor = (mode) => {
    return getStrategyInfo(mode).color;
  };
  
  if (!user?.has_api_token) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="text-center">
          <div className="text-gray-400 text-6xl mb-4">🤖</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">AI Trading Bot</h3>
          <p className="text-gray-600 mb-4">
            Please configure your Deriv API token to activate the trading bot
          </p>
          <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
            Setup API Token
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Main Bot Control */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${botStatus?.is_active ? 'bg-green-500' : 'bg-gray-400'}`}></div>
            <h2 className="text-xl font-semibold text-gray-900">AI Trading Bot</h2>
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              botStatus?.is_active ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
            }`}>
              {botStatus?.is_active ? 'Active' : 'Inactive'}
            </span>
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={() => setShowSettings(true)}
              className="bg-gray-100 text-gray-700 px-3 py-2 rounded-md hover:bg-gray-200"
            >
              ⚙️ Settings
            </button>
            
            {botStatus?.is_active ? (
              <button
                onClick={stopBot}
                disabled={isLoading}
                className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 disabled:opacity-50"
              >
                {isLoading ? 'Stopping...' : 'Stop Bot'}
              </button>
            ) : (
              <button
                onClick={startBot}
                disabled={isLoading}
                className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 disabled:opacity-50"
              >
                {isLoading ? 'Starting...' : 'Start Bot'}
              </button>
            )}
          </div>
        </div>
        
        {/* Bot Status Display */}
        {botStatus && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Current Strategy</div>
              <div className="flex items-center space-x-2 mt-1">
                {botStatus.current_mode ? (
                  <>
                    <span className="text-lg">{getStrategyInfo(botStatus.current_mode).icon}</span>
                    <div className="flex-1">
                      <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        getStrategyInfo(botStatus.current_mode).bgColor
                      } ${getStrategyInfo(botStatus.current_mode).textColor}`}>
                        {getStrategyInfo(botStatus.current_mode).name}
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {getStrategyInfo(botStatus.current_mode).description}
                      </p>
                    </div>
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 rounded-full bg-gray-300 animate-pulse"></span>
                    <span className="font-medium text-gray-400">Initializing...</span>
                  </>
                )}
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Account Balance</div>
              <div className="text-lg font-semibold">${botStatus.account_balance?.toFixed(2)}</div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Daily P&L</div>
              <div className={`text-lg font-semibold ${
                botStatus.daily_pnl >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {botStatus.daily_pnl >= 0 ? '+' : ''}${botStatus.daily_pnl?.toFixed(2)}
              </div>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-sm text-gray-600">Win Rate</div>
              <div className="text-lg font-semibold">{botStatus.win_rate?.toFixed(1)}%</div>
            </div>
          </div>
        )}
        
        {/* Strategy Performance */}
        {botStatus?.strategy_performance && (
          <div className="border-t pt-4">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Strategy Performance</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {Object.entries(botStatus.strategy_performance).map(([strategy, performance]) => (
                <div key={strategy} className="bg-gray-50 p-3 rounded">
                  <div className="text-xs text-gray-600">{strategy.replace('_', ' ')}</div>
                  <div className="text-sm font-medium">
                    {performance.wins}W / {performance.losses}L
                  </div>
                  <div className={`text-xs ${performance.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${performance.profit?.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {/* Safety Status */}
        {botStatus && (
          <div className="border-t pt-4 mt-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Safety Status</span>
              <div className="flex space-x-4 text-sm">
                <span className={`${botStatus.in_cooldown ? 'text-orange-600' : 'text-gray-600'}`}>
                  {botStatus.in_cooldown ? `Cooldown (${botStatus.cooldown_trades_left} left)` : 'Active'}
                </span>
                <span className="text-gray-600">
                  Martingale: {botStatus.martingale_step}/3
                </span>
              </div>
            </div>
            
            <div className="mt-2 grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Daily Target: </span>
                <span className="font-medium">${botStatus.daily_target?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-gray-600">Max Loss: </span>
                <span className="font-medium">${botStatus.max_daily_loss?.toFixed(2)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
      
      {/* Action Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          onClick={() => {
            setShowPerformance(true);
            fetchPerformanceData();
          }}
          className="bg-blue-50 border border-blue-200 p-4 rounded-lg hover:bg-blue-100 transition-colors"
        >
          <div className="text-blue-600 text-2xl mb-2">📊</div>
          <div className="font-medium text-blue-900">Performance Analytics</div>
          <div className="text-sm text-blue-600">View detailed statistics</div>
        </button>
        
        <button
          onClick={() => {
            setShowHistory(true);
            fetchTradingHistory();
          }}
          className="bg-green-50 border border-green-200 p-4 rounded-lg hover:bg-green-100 transition-colors"
        >
          <div className="text-green-600 text-2xl mb-2">📈</div>
          <div className="font-medium text-green-900">Trading History</div>
          <div className="text-sm text-green-600">View trade records</div>
        </button>
        
        <button
          onClick={() => {
            fetchMlModelPerformance();
            retrainModels();
          }}
          className="bg-purple-50 border border-purple-200 p-4 rounded-lg hover:bg-purple-100 transition-colors"
        >
          <div className="text-purple-600 text-2xl mb-2">🧠</div>
          <div className="font-medium text-purple-900">AI Models</div>
          <div className="text-sm text-purple-600">Retrain & optimize</div>
        </button>
      </div>
      
      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Bot Settings</h3>
              <button
                onClick={() => setShowSettings(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            
            <div className="space-y-4">
              {/* Risk Management */}
              <div>
                <h4 className="font-medium mb-3">Risk Management</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Daily Stop Loss (%)
                    </label>
                    <input
                      type="number"
                      value={botSettings.daily_stop_loss_percent}
                      onChange={(e) => handleSettingChange('daily_stop_loss_percent', parseFloat(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="1"
                      max="50"
                      step="0.1"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Daily Target (%)
                    </label>
                    <input
                      type="number"
                      value={botSettings.daily_target_percent}
                      onChange={(e) => handleSettingChange('daily_target_percent', parseFloat(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="1"
                      max="100"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>
              
              {/* Money Management */}
              <div>
                <h4 className="font-medium mb-3">Money Management</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Base Stake (%)
                    </label>
                    <input
                      type="number"
                      value={botSettings.base_stake_percent}
                      onChange={(e) => handleSettingChange('base_stake_percent', parseFloat(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="0.1"
                      max="10"
                      step="0.1"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Max Stake (%)
                    </label>
                    <input
                      type="number"
                      value={botSettings.max_stake_percent}
                      onChange={(e) => handleSettingChange('max_stake_percent', parseFloat(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="1"
                      max="25"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>
              
              {/* Martingale Settings */}
              <div>
                <h4 className="font-medium mb-3">Smart Martingale</h4>
                <div className="flex items-center mb-3">
                  <input
                    type="checkbox"
                    id="enable_martingale"
                    checked={botSettings.enable_martingale}
                    onChange={(e) => handleSettingChange('enable_martingale', e.target.checked)}
                    className="mr-2"
                  />
                  <label htmlFor="enable_martingale" className="text-sm">Enable Smart Martingale</label>
                </div>
                
                {botSettings.enable_martingale && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Multiplier
                      </label>
                      <input
                        type="number"
                        value={botSettings.martingale_multiplier}
                        onChange={(e) => handleSettingChange('martingale_multiplier', parseFloat(e.target.value))}
                        className="w-full border border-gray-300 rounded-md px-3 py-2"
                        min="1.1"
                        max="3.0"
                        step="0.1"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Max Steps
                      </label>
                      <input
                        type="number"
                        value={botSettings.max_martingale_steps}
                        onChange={(e) => handleSettingChange('max_martingale_steps', parseInt(e.target.value))}
                        className="w-full border border-gray-300 rounded-md px-3 py-2"
                        min="1"
                        max="5"
                      />
                    </div>
                  </div>
                )}
              </div>
              
              {/* Strategy Settings */}
              <div>
                <h4 className="font-medium mb-3">Strategy Control</h4>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Switch After Wins
                    </label>
                    <input
                      type="number"
                      value={botSettings.strategy_switch_wins}
                      onChange={(e) => handleSettingChange('strategy_switch_wins', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="1"
                      max="10"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Switch After Losses
                    </label>
                    <input
                      type="number"
                      value={botSettings.strategy_switch_losses}
                      onChange={(e) => handleSettingChange('strategy_switch_losses', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="1"
                      max="5"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Re-evaluate Every
                    </label>
                    <input
                      type="number"
                      value={botSettings.reevaluate_trades}
                      onChange={(e) => handleSettingChange('reevaluate_trades', parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded-md px-3 py-2"
                      min="5"
                      max="50"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={updateSettings}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Performance Analytics Modal */}
      {showPerformance && performanceData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Performance Analytics</h3>
              <button
                onClick={() => setShowPerformance(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            
            {/* Strategy Performance */}
            <div className="mb-6">
              <h4 className="font-medium mb-3">Strategy Performance</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {performanceData.strategy_performance.map((strategy, index) => (
                  <div key={index} className="border rounded-lg p-4">
                    <div className="font-medium">{strategy.strategy.replace('_', ' ')}</div>
                    <div className="text-sm text-gray-600 mt-1">
                      {strategy.total_trades} trades, {strategy.win_rate.toFixed(1)}% win rate
                    </div>
                    <div className={`text-lg font-semibold mt-2 ${
                      strategy.total_profit >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ${strategy.total_profit.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Contract Performance */}
            <div className="mb-6">
              <h4 className="font-medium mb-3">Contract Type Performance</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {performanceData.contract_performance.map((contract, index) => (
                  <div key={index} className="border rounded-lg p-3">
                    <div className="font-medium text-sm">{contract.contract_type.replace('_', ' ')}</div>
                    <div className="text-xs text-gray-600">
                      {contract.total_trades} trades, {contract.win_rate.toFixed(1)}% win rate
                    </div>
                    <div className={`text-sm font-semibold ${
                      contract.total_profit >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ${contract.total_profit.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Daily Performance */}
            <div>
              <h4 className="font-medium mb-3">Daily Performance (Last 30 Days)</h4>
              <div className="max-h-64 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-2 text-left">Date</th>
                      <th className="px-3 py-2 text-left">Trades</th>
                      <th className="px-3 py-2 text-left">Win Rate</th>
                      <th className="px-3 py-2 text-left">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performanceData.daily_performance.map((day, index) => (
                      <tr key={index} className="border-b">
                        <td className="px-3 py-2">{day.date}</td>
                        <td className="px-3 py-2">{day.trades}</td>
                        <td className="px-3 py-2">{day.win_rate.toFixed(1)}%</td>
                        <td className={`px-3 py-2 ${
                          day.daily_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          ${day.daily_pnl.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
      
      {/* Trading History Modal */}
      {showHistory && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-6xl max-h-[90vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">Trading History</h3>
              <button
                onClick={() => setShowHistory(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-2 text-left">Time</th>
                    <th className="px-3 py-2 text-left">Contract</th>
                    <th className="px-3 py-2 text-left">Strategy</th>
                    <th className="px-3 py-2 text-left">Entry</th>
                    <th className="px-3 py-2 text-left">Exit</th>
                    <th className="px-3 py-2 text-left">Stake</th>
                    <th className="px-3 py-2 text-left">P&L</th>
                    <th className="px-3 py-2 text-left">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {tradingHistory.map((trade, index) => (
                    <tr key={index} className="border-b">
                      <td className="px-3 py-2">
                        {new Date(trade.timestamp).toLocaleString()}
                      </td>
                      <td className="px-3 py-2">
                        {trade.contract_type.replace('_', ' ')}
                      </td>
                      <td className="px-3 py-2">
                        {trade.strategy_used.replace('_', ' ')}
                      </td>
                      <td className="px-3 py-2">
                        {trade.entry_price.toFixed(5)}
                      </td>
                      <td className="px-3 py-2">
                        {trade.exit_price.toFixed(5)}
                      </td>
                      <td className="px-3 py-2">
                        ${trade.stake_amount.toFixed(2)}
                      </td>
                      <td className={`px-3 py-2 ${
                        trade.profit_loss >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        ${trade.profit_loss.toFixed(2)}
                      </td>
                      <td className="px-3 py-2">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          trade.success 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {trade.success ? 'Win' : 'Loss'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingBot;
