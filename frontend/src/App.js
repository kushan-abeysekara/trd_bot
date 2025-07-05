import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import io from 'socket.io-client';
import './App.css';

function App() {
  const [apiToken, setApiToken] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isTrading, setIsTrading] = useState(false);
  const [balance, setBalance] = useState(0);
  const balanceRef = useRef(0); // Keep a ref for balance comparisons
  const [tradeAmount, setTradeAmount] = useState(0.35); // Minimum allowed trade amount
  // Duration is now dynamic based on the strategy signal
  const [stats, setStats] = useState({
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    total_profit_loss: 0,
    winning_rate: 0
  });
  const [tradeHistory, setTradeHistory] = useState([]);
  const [socket, setSocket] = useState(null);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  
  // Strategy-related state
  const [strategies, setStrategies] = useState([]);
  const [indicators, setIndicators] = useState({});
  const [lastSignal, setLastSignal] = useState(null);
  const [tradingMode, setTradingMode] = useState('strategy');

  // Take Profit and Stop Loss state
  const [takeProfitEnabled, setTakeProfitEnabled] = useState(false);
  const [takeProfitAmount, setTakeProfitAmount] = useState(0);
  const [stopLossEnabled, setStopLossEnabled] = useState(false);
  const [stopLossAmount, setStopLossAmount] = useState(0);
  const [sessionStats, setSessionStats] = useState({
    session_profit_loss: 0,
    initial_balance: 0,
    current_balance: 0
  });

  useEffect(() => {
    // Dynamically determine the API URL based on current environment
    const getApiUrl = () => {
      const hostname = window.location.hostname;
      if (hostname === 'localhost' || hostname === '127.0.0.1') {
        return 'http://localhost:5000';
      } else {
        return 'https://tradingbot-4iuxi.ondigitalocean.app';
      }
    };

    const apiUrl = getApiUrl();
    console.log(`🌐 Connecting to API at: ${apiUrl}`);

    // Configure axios defaults
    axios.defaults.baseURL = apiUrl;
    
    // Initialize socket connection
    const newSocket = io(apiUrl, {
      withCredentials: true,
      transports: ['websocket', 'polling']
    });
    setSocket(newSocket);

    newSocket.on('connection_status', (data) => {
      setIsConnected(data.connected);
      if (data.connected) {
        setBalance(data.balance || 0);
        setMessage('Successfully connected to Deriv API');
        setMessageType('success');
        
        // Load strategies and indicators when connected
        loadStrategies();
        loadIndicators();
        loadSessionStats();
      } else {
        setMessage(data.error || 'Failed to connect to Deriv API');
        setMessageType('error');
      }
    });

    newSocket.on('balance_update', (data) => {
      const newBalance = data.balance;
      const oldBalance = balanceRef.current;
      
      setBalance(newBalance);
      balanceRef.current = newBalance;
      
      // Show balance change notification only if it's a significant change (not initial load)
      if (oldBalance > 0 && Math.abs(newBalance - oldBalance) > 0.01) {
        const change = newBalance - oldBalance;
        const changeText = change > 0 ? `+${formatCurrency(change)}` : formatCurrency(change);
        setMessage(`💰 Balance updated: ${changeText} → ${formatCurrency(newBalance)}`);
        setMessageType(change > 0 ? 'success' : 'error');
        
        // Clear message after 4 seconds
        setTimeout(() => {
          setMessage('');
        }, 4000);
      }
    });

    newSocket.on('trade_update', (trade) => {
      setTradeHistory(prev => {
        const existingIndex = prev.findIndex(t => t.id === trade.id);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = trade;
          return updated;
        } else {
          return [trade, ...prev];
        }
      });
    });

    newSocket.on('stats_update', (newStats) => {
      setStats(newStats);
      // Update session stats if available in the stats update
      if (newStats.initial_balance !== undefined && newStats.session_profit_loss !== undefined) {
        setSessionStats(prev => ({
          ...prev,
          initial_balance: newStats.initial_balance,
          session_profit_loss: newStats.session_profit_loss
        }));
      }
    });

    newSocket.on('strategy_signal', (signalData) => {
      setLastSignal(signalData);
      setIndicators(signalData.indicators || {});
      
      // Show strategy signal notification
      setMessage(`🎯 ${signalData.strategy_name} - ${signalData.direction} (${(signalData.confidence * 100).toFixed(0)}% confidence)`);
      setMessageType('info');
      
      // Clear message after 5 seconds
      setTimeout(() => {
        setMessage('');
      }, 5000);
    });

    // Handle session stats updates
    newSocket.on('session_stats_update', (stats) => {
      setSessionStats(stats);
      setTakeProfitEnabled(stats.take_profit_enabled);
      setTakeProfitAmount(stats.take_profit_amount);
      setStopLossEnabled(stats.stop_loss_enabled);
      setStopLossAmount(stats.stop_loss_amount);
    });

    // Listen for take profit/stop loss notifications
    newSocket.on('trading_stopped', (data) => {
      setIsTrading(false);
      if (data.reason === 'take_profit') {
        setMessage(`🎉 Take Profit Hit! Trading stopped. Profit: ${formatCurrency(data.amount)}`);
        setMessageType('success');
      } else if (data.reason === 'stop_loss') {
        setMessage(`⚠️ Stop Loss Hit! Trading stopped. Loss: ${formatCurrency(data.amount)}`);
        setMessageType('error');
      }
    });

    // Set up periodic session stats updates
    const statsInterval = setInterval(() => {
      if (isConnected) {
        loadSessionStats();
      }
    }, 5000); // Update every 5 seconds

    return () => {
      newSocket.close();
      clearInterval(statsInterval);
    };
  }, [isConnected]);

  // Load available strategies
  const loadStrategies = async () => {
    try {
      const response = await axios.get('/api/strategies');
      setStrategies(response.data.strategies);
    } catch (error) {
      console.error('Failed to load strategies:', error);
    }
  };

  // Load current indicators
  const loadIndicators = async () => {
    try {
      const response = await axios.get('/api/indicators');
      setIndicators(response.data);
    } catch (error) {
      console.error('Failed to load indicators:', error);
    }
  };

  // Load session stats
  const loadSessionStats = async () => {
    try {
      const response = await axios.get('/api/session-stats');
      setSessionStats(response.data);
      setTakeProfitEnabled(response.data.take_profit_enabled);
      setTakeProfitAmount(response.data.take_profit_amount);
      setStopLossEnabled(response.data.stop_loss_enabled);
      setStopLossAmount(response.data.stop_loss_amount);
    } catch (error) {
      console.error('Failed to load session stats:', error);
    }
  };

  // Set take profit
  const handleSetTakeProfit = async (enabled, amount) => {
    try {
      await axios.post('/api/set-take-profit', { enabled, amount });
      setTakeProfitEnabled(enabled);
      setTakeProfitAmount(amount);
      setMessage(`Take profit ${enabled ? 'enabled' : 'disabled'}`);
      setMessageType('success');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to set take profit');
      setMessageType('error');
    }
  };

  // Set stop loss
  const handleSetStopLoss = async (enabled, amount) => {
    try {
      await axios.post('/api/set-stop-loss', { enabled, amount });
      setStopLossEnabled(enabled);
      setStopLossAmount(amount);
      setMessage(`Stop loss ${enabled ? 'enabled' : 'disabled'}`);
      setMessageType('success');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to set stop loss');
      setMessageType('error');
    }
  };

  // Set trading mode
  const handleTradingModeChange = async (mode) => {
    try {
      await axios.post('/api/trading-mode', { mode });
      setTradingMode(mode);
      setMessage(`Trading mode set to ${mode}`);
      setMessageType('success');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to set trading mode');
      setMessageType('error');
    }
  };

  // Refresh balance manually
  const handleRefreshBalance = async () => {
    try {
      const response = await axios.post('/api/refresh-balance');
      setBalance(response.data.balance);
      setMessage('Balance refreshed successfully');
      setMessageType('success');
      
      // Clear message after 3 seconds
      setTimeout(() => {
        setMessage('');
      }, 8080);
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to refresh balance');
      setMessageType('error');
    }
  };

  const handleConnect = async () => {
    if (!apiToken.trim()) {
      setMessage('Please enter your API token');
      setMessageType('error');
      return;
    }

    try {
      setMessage('Connecting to Deriv API...');
      setMessageType('info');
      
      await axios.post('/api/connect', {
        api_token: apiToken
      });
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to connect');
      setMessageType('error');
    }
  };

  const handleStartTrading = async () => {
    try {
      await axios.post('/api/start-trading', {
        amount: tradeAmount,
        // Duration is now handled dynamically in the backend based on each signal
      });
      setIsTrading(true);
      setMessage('Trading started successfully');
      setMessageType('success');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to start trading');
      setMessageType('error');
    }
  };

  const handleStopTrading = async () => {
    try {
      await axios.post('/api/stop-trading');
      setIsTrading(false);
      setMessage('Trading stopped');
      setMessageType('success');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to stop trading');
      setMessageType('error');
    }
  };

  const handleDisconnect = async () => {
    try {
      await axios.post('/api/disconnect');
      setIsConnected(false);
      setIsTrading(false);
      setBalance(0);
      setTradeHistory([]);
      setStats({
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        total_profit_loss: 0,
        winning_rate: 0
      });
      setMessage('Disconnected from Deriv API');
      setMessageType('success');
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to disconnect');
      setMessageType('error');
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🤖 Deriv Volatility Index Trading Bot</h1>
        <p>Automated trading for Volatility 100 Index with Rise/Fall contracts</p>
      </div>

      {message && (
        <div className={`${messageType === 'error' ? 'error-message' : 'success-message'}`}>
          {message}
        </div>
      )}

      <div className="main-content">
        <div className="card">
          <h2>API Connection</h2>
          
          <div className="status-indicator">
            <div className={`status-dot ${isConnected ? 'status-connected' : 'status-disconnected'}`}></div>
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>

          {!isConnected ? (
            <div className="api-section">
              <div className="form-group">
                <label htmlFor="apiToken">Deriv API Token:</label>
                <input
                  type="text"
                  id="apiToken"
                  value={apiToken}
                  onChange={(e) => setApiToken(e.target.value)}
                  placeholder="Enter your Deriv API token here"
                />
              </div>
              <button 
                className="btn btn-primary" 
                onClick={handleConnect}
                disabled={!apiToken.trim()}
              >
                Connect to Deriv API
              </button>
            </div>
          ) : (
            <div className="api-section">
              <div className="balance-display">
                Balance: {formatCurrency(balance)}
                <button 
                  className="btn btn-secondary btn-small"
                  onClick={handleRefreshBalance}
                  style={{ marginLeft: '15px', fontSize: '0.8rem', padding: '5px 10px' }}
                  title="Refresh balance from server"
                >
                  🔄 Refresh
                </button>
              </div>
              
              <div className="form-group">
                <label htmlFor="tradeAmount">Trade Amount (USD):</label>
                <input
                  type="number"
                  id="tradeAmount"
                  value={tradeAmount}
                  onChange={(e) => setTradeAmount(parseFloat(e.target.value) || 0.35)}
                  min="0.35"
                  step="0.01"
                  disabled={isTrading}
                />
              </div>

              {/* Duration dropdown removed - now using dynamic duration based on signal */}

              {/* Take Profit Settings */}
              <div className="form-group">
                <div className="checkbox-group">
                  <input
                    type="checkbox"
                    id="takeProfitEnabled"
                    checked={takeProfitEnabled}
                    onChange={(e) => {
                      const enabled = e.target.checked;
                      setTakeProfitEnabled(enabled);
                      if (!enabled) {
                        handleSetTakeProfit(false, 0);
                      }
                    }}
                    disabled={isTrading}
                  />
                  <label htmlFor="takeProfitEnabled">Enable Take Profit</label>
                </div>
                {takeProfitEnabled && (
                  <div className="input-with-button">
                    <input
                      type="number"
                      value={takeProfitAmount}
                      onChange={(e) => setTakeProfitAmount(parseFloat(e.target.value) || 0)}
                      placeholder="Take profit amount (USD)"
                      min="0"
                      step="0.1"
                      disabled={isTrading}
                    />
                    <button 
                      className="btn btn-secondary btn-small"
                      onClick={() => handleSetTakeProfit(true, takeProfitAmount)}
                      disabled={isTrading}
                    >
                      Set
                    </button>
                  </div>
                )}
              </div>

              {/* Stop Loss Settings */}
              <div className="form-group">
                <div className="checkbox-group">
                  <input
                    type="checkbox"
                    id="stopLossEnabled"
                    checked={stopLossEnabled}
                    onChange={(e) => {
                      const enabled = e.target.checked;
                      setStopLossEnabled(enabled);
                      if (!enabled) {
                        handleSetStopLoss(false, 0);
                      }
                    }}
                    disabled={isTrading}
                  />
                  <label htmlFor="stopLossEnabled">Enable Stop Loss</label>
                </div>
                {stopLossEnabled && (
                  <div className="input-with-button">
                    <input
                      type="number"
                      value={stopLossAmount}
                      onChange={(e) => setStopLossAmount(parseFloat(e.target.value) || 0)}
                      placeholder="Stop loss amount (USD)"
                      min="0"
                      step="0.1"
                      disabled={isTrading}
                    />
                    <button 
                      className="btn btn-secondary btn-small"
                      onClick={() => handleSetStopLoss(true, stopLossAmount)}
                      disabled={isTrading}
                    >
                      Set
                    </button>
                  </div>
                )}
              </div>

              {/* Session Stats Display */}
              {(takeProfitEnabled || stopLossEnabled) && (
                <div className="session-stats">
                  <h4>💰 Session Statistics</h4>
                  <div className="stats-row">
                    <span>Session P&L: </span>
                    <span className={sessionStats.session_profit_loss >= 0 ? 'profit' : 'loss'}>
                      {formatCurrency(sessionStats.session_profit_loss)}
                    </span>
                  </div>
                  <div className="stats-row">
                    <span>Initial Balance: </span>
                    <span>{formatCurrency(sessionStats.initial_balance)}</span>
                  </div>
                  {takeProfitEnabled && (
                    <div className="stats-row">
                      <span>Take Profit Target: </span>
                      <span className="profit">{formatCurrency(takeProfitAmount)}</span>
                    </div>
                  )}
                  {stopLossEnabled && (
                    <div className="stats-row">
                      <span>Stop Loss Limit: </span>
                      <span className="loss">-{formatCurrency(stopLossAmount)}</span>
                    </div>
                  )}
                </div>
              )}

              <div style={{ marginBottom: '20px' }}>
                {!isTrading ? (
                  <button 
                    className="btn btn-success" 
                    onClick={handleStartTrading}
                  >
                    🚀 Start Trading
                  </button>
                ) : (
                  <button 
                    className="btn btn-danger" 
                    onClick={handleStopTrading}
                  >
                    ⏹️ Stop Trading
                  </button>
                )}
                
                <button 
                  className="btn btn-primary" 
                  onClick={handleDisconnect}
                >
                  Disconnect
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Trading Statistics</h2>
          
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">{stats.total_trades}</div>
              <div className="stat-label">Total Trades</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{stats.winning_rate.toFixed(1)}%</div>
              <div className="stat-label">Win Rate</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{stats.winning_trades}</div>
              <div className="stat-label">Wins</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{stats.losing_trades}</div>
              <div className="stat-label">Losses</div>
            </div>
          </div>

          <div className="balance-display" style={{ fontSize: '1.5rem', marginTop: '20px' }}>
            Total P&L: <span className={stats.total_profit_loss >= 0 ? 'profit' : 'loss'}>
              {formatCurrency(stats.total_profit_loss)}
            </span>
          </div>

          {stats.initial_balance > 0 && (
            <div className="balance-details">
              <div className="stats-row">
                <span>Starting Balance:</span>
                <span>{formatCurrency(stats.initial_balance)}</span>
              </div>
              <div className="stats-row">
                <span>Session P&L:</span>
                <span className={stats.session_profit_loss >= 0 ? 'profit' : 'loss'}>
                  {formatCurrency(stats.session_profit_loss)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Strategy Information Section */}
      {isConnected && (
        <div className="main-content">
          <div className="card">
            <h2>🧠 Strategy Engine</h2>
            
            <div className="form-group">
              <label>Trading Mode:</label>
              <select 
                value={tradingMode} 
                onChange={(e) => handleTradingModeChange(e.target.value)}
                disabled={isTrading}
              >
                <option value="strategy">Strategy-Based Trading</option>
                <option value="random">Random Trading</option>
              </select>
            </div>

            {lastSignal && (
              <div className="strategy-signal">
                <h3>🎯 Latest Strategy Signal</h3>
                <div className="signal-info">
                  <div className="signal-header">
                    <span className="strategy-name">{lastSignal.strategy_name}</span>
                    <span className={`signal-direction ${lastSignal.direction.toLowerCase()}`}>
                      {lastSignal.direction === 'CALL' ? '📈 RISE' : '📉 FALL'}
                    </span>
                  </div>
                  <div className="signal-details">
                    <div>Confidence: {(lastSignal.confidence * 100).toFixed(0)}%</div>
                    <div>Hold Time: {lastSignal.hold_time}s</div>
                  </div>
                  <div className="signal-reason">{lastSignal.entry_reason}</div>
                  <div className="signal-conditions">
                    <strong>Conditions Met:</strong>
                    <ul>
                      {lastSignal.conditions_met.map((condition, index) => (
                        <li key={index}>{condition}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="card">
            <h2>📊 Technical Indicators</h2>
            
            {Object.keys(indicators).length > 0 ? (
              <div className="indicators-grid">
                <div className="indicator-item">
                  <div className="indicator-label">RSI</div>
                  <div className={`indicator-value ${indicators.rsi > 70 ? 'overbought' : indicators.rsi < 30 ? 'oversold' : 'neutral'}`}>
                    {indicators.rsi?.toFixed(1)}
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">MACD</div>
                  <div className={`indicator-value ${indicators.macd > 0 ? 'bullish' : 'bearish'}`}>
                    {indicators.macd?.toFixed(4)}
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">Momentum</div>
                  <div className={`indicator-value ${indicators.momentum > 0 ? 'bullish' : 'bearish'}`}>
                    {indicators.momentum?.toFixed(2)}%
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">Volatility</div>
                  <div className={`indicator-value ${indicators.volatility > 2 ? 'high' : indicators.volatility < 0.5 ? 'low' : 'normal'}`}>
                    {indicators.volatility?.toFixed(2)}%
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">BB Upper</div>
                  <div className="indicator-value">{indicators.bb_upper?.toFixed(5)}</div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">BB Lower</div>
                  <div className="indicator-value">{indicators.bb_lower?.toFixed(5)}</div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">EMA5</div>
                  <div className="indicator-value">{indicators.ema5?.toFixed(5)}</div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">Tick Count</div>
                  <div className="indicator-value">{indicators.tick_count}</div>
                </div>
              </div>
            ) : (
              <p style={{ textAlign: 'center', color: '#666', marginTop: '20px' }}>
                Indicators will appear here once trading starts...
              </p>
            )}
          </div>
        </div>
      )}

      <div className="card trade-history">
        <h2>Trade History</h2>
        
        {tradeHistory.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#666', marginTop: '20px' }}>
            No trades yet. Start trading to see your trade history here.
          </p>
        ) : (
          <div className="trade-list">
            {tradeHistory.map((trade, index) => (
              <div key={trade.id || index} className="trade-item">
                <div className="trade-info">
                  <div className={`trade-type ${trade.type.toLowerCase()}`}>
                    {trade.type === 'CALL' ? '📈 RISE' : '📉 FALL'}
                  </div>
                  <div className="trade-details">
                    {formatTime(trade.timestamp)} • {formatCurrency(trade.amount)} • {trade.duration} ticks
                  </div>
                  {trade.strategy_name && (
                    <div className="strategy-info">
                      <div className="strategy-badge">🧠 {trade.strategy_name}</div>
                      {trade.strategy_confidence && (
                        <div className="confidence-badge">
                          {(trade.strategy_confidence * 100).toFixed(0)}% confidence
                        </div>
                      )}
                    </div>
                  )}
                  {trade.entry_reason && (
                    <div className="entry-reason">{trade.entry_reason}</div>
                  )}
                </div>
                
                <div className="trade-result">
                  <div className={`trade-status status-${trade.result || 'active'}`}>
                    {trade.status === 'active' ? 'ACTIVE' : (trade.result === 'win' ? 'WIN' : 'LOSS')}
                  </div>
                  {trade.profit_loss !== undefined && (
                    <div className={`profit-loss ${trade.profit_loss >= 0 ? 'profit' : 'loss'}`}>
                      {trade.profit_loss >= 0 ? '+' : ''}{formatCurrency(trade.profit_loss)}
                    </div>
                  )}
                  {trade.actual_win_probability && (
                    <div className="win-probability">
                      {(trade.actual_win_probability * 100).toFixed(0)}% win chance
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
