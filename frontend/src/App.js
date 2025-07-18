import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

// Configure API base URL based on environment
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

// Configure axios defaults
axios.defaults.baseURL = API_BASE_URL;

// Polling interval in milliseconds
const POLLING_INTERVAL = 2000; // 2 seconds
const HEARTBEAT_INTERVAL = 15000; // 15 seconds

function App() {
  const [apiToken, setApiToken] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isTrading, setIsTrading] = useState(false);
  const [balance, setBalance] = useState(0);
  const balanceRef = useRef(0); // Keep a ref for balance comparisons
  const initialBalanceRef = useRef(0); // Add a ref for tracking initial balance
  const [tradeAmount, setTradeAmount] = useState(1);
  const [stats, setStats] = useState({
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    total_profit_loss: 0,
    winning_rate: 0
  });
  const [tradeHistory, setTradeHistory] = useState([]);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');
  
  // Strategy-related state
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
  const [initialBalance, setInitialBalance] = useState(0);
  const [realProfit, setRealProfit] = useState(0);
  
  // Polling references
  const pollingRef = useRef(null);
  const heartbeatRef = useRef(null);

  // Check connection status on initial load
  useEffect(() => {
    checkConnectionStatus();
  }, []);

  // Setup polling when connected
  useEffect(() => {
    if (isConnected) {
      startPolling();
      startHeartbeat();
      return () => {
        stopPolling();
        stopHeartbeat();
      }
    }
  }, [isConnected]);

  // Check if already connected when component mounts
  const checkConnectionStatus = async () => {
    try {
      const response = await axios.get('/api/connection-status');
      if (response.data.connected) {
        setIsConnected(true);
        setBalance(response.data.balance || 0);
        balanceRef.current = response.data.balance || 0;
        
        // Properly set initial balance
        if (response.data.initial_balance && response.data.initial_balance > 0) {
          setInitialBalance(response.data.initial_balance);
          initialBalanceRef.current = response.data.initial_balance;
          // Calculate real profit based on initial balance
          const calculatedProfit = (response.data.balance || 0) - response.data.initial_balance;
          setRealProfit(calculatedProfit);
        }
        setMessage('Successfully connected to Deriv API');
        setMessageType('success');
        
        // Load initial data
        fetchStats();
        fetchTradeHistory();
        fetchSessionStats();
        fetchIndicators();
        fetchLatestSignal();
        
        // Start heartbeat immediately
        startHeartbeat();
      }
    } catch (error) {
      console.error('Error checking connection status:', error);
    }
  };

  // Start heartbeat to keep connection alive
  const startHeartbeat = () => {
    if (heartbeatRef.current) return;
    
    // Send a heartbeat every HEARTBEAT_INTERVAL
    const sendHeartbeat = async () => {
      if (!isConnected) return;
      
      try {
        const response = await axios.post('/api/heartbeat');
        if (!response.data.connected) {
          // Connection lost, attempt to reconnect
          checkConnectionStatus();
        }
      } catch (error) {
        console.error('Heartbeat error:', error);
      }
    };
    
    // Send initial heartbeat
    sendHeartbeat();
    
    // Setup heartbeat interval
    heartbeatRef.current = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL);
  };
  
  // Stop heartbeat
  const stopHeartbeat = () => {
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current);
      heartbeatRef.current = null;
    }
  };

  // Add a safe toFixed method to handle undefined values
  const safeToFixed = (value, digits = 1) => {
    if (value === undefined || value === null) {
      return '0.0';
    }
    return value.toFixed(digits);
  };

  // Start polling for updates
  const startPolling = () => {
    if (pollingRef.current) return;
    
    // Function to fetch all updates at once
    const fetchUpdates = async () => {
      try {
        const response = await axios.get('/api/updates');
        const data = response.data;
        
        // Update all states from response
        setIsConnected(data.connected);
        setIsTrading(data.trading);
        
        // Update balance with notification if changed
        if (data.balance !== balanceRef.current) {
          updateBalance(data.balance);
        }
        
        // Update other states
        setStats(data.stats || stats);
        
        // FORCE update trade history from full data
        if (data.trade_history_full && data.trade_history_full.length >= 0) {
          setTradeHistory(data.trade_history_full);
        }
        // Fallback to recent trades if full history not available
        else if (data.recent_trades && data.recent_trades.length > 0) {
          updateTradeHistory(data.recent_trades);
        }
        
        // Update indicators
        setIndicators(data.indicators || {});
        
        // Update session stats
        setSessionStats(data.session_stats || sessionStats);
        
        // Update latest signal if available
        if (data.latest_signal && (!lastSignal || 
            data.latest_signal.timestamp > lastSignal.timestamp)) {
          setLastSignal(data.latest_signal);
          showSignalNotification(data.latest_signal);
        }
        
        // Update initial balance if provided and not set yet
        if (data.initial_balance && data.initial_balance > 0) {
          if (initialBalanceRef.current === 0) {
            setInitialBalance(data.initial_balance);
            initialBalanceRef.current = data.initial_balance;
            console.log("Initial balance set to:", data.initial_balance);
          }
        }
        
        // Calculate real profit - ensure we have a valid initial balance first
        if (data.balance && initialBalanceRef.current > 0) {
          const calculatedProfit = data.balance - initialBalanceRef.current;
          setRealProfit(calculatedProfit);
        }
        
        // Debug trade status counts
        if (data.trade_count_debug) {
          console.log(`📊 Trade Status: ${data.trade_count_debug.active} active, ${data.trade_count_debug.completed} completed`);
        }
        
      } catch (error) {
        console.error('Polling error:', error);
        
        // On polling error, try to refresh trade history separately
        try {
          const tradeResponse = await axios.get('/api/trade-history');
          if (tradeResponse.data.trades) {
            setTradeHistory(tradeResponse.data.trades);
            console.log('✅ Trade history refreshed after polling error');
          }
        } catch (tradeError) {
          console.error('Trade history refresh error:', tradeError);
        }
      }
    };
    
    // Initial fetch
    fetchUpdates();
    
    // Setup polling interval - increased frequency for better trade updates
    pollingRef.current = setInterval(fetchUpdates, 1500); // 1.5 seconds
  };

  // Stop polling
  const stopPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  // Fetch stats separately
  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  // Fetch trade history
  const fetchTradeHistory = async () => {
    try {
      const response = await axios.get('/api/trade-history');
      if (response.data.trades) {
        setTradeHistory(response.data.trades);
        
        // Debug log
        const { active_count, completed_count, total_count } = response.data;
        console.log(`📋 Trade History: ${total_count} total (${active_count} active, ${completed_count} completed)`);
      }
    } catch (error) {
      console.error('Error fetching trade history:', error);
    }
  };

  // Fetch session stats
  const fetchSessionStats = async () => {
    try {
      const response = await axios.get('/api/session-stats');
      setSessionStats(response.data);
      setTakeProfitEnabled(response.data.take_profit_enabled);
      setTakeProfitAmount(response.data.take_profit_amount);
      setStopLossEnabled(response.data.stop_loss_enabled);
      setStopLossAmount(response.data.stop_loss_amount);
    } catch (error) {
      console.error('Error fetching session stats:', error);
    }
  };

  // Fetch indicators
  const fetchIndicators = async () => {
    try {
      const response = await axios.get('/api/indicators');
      setIndicators(response.data);
    } catch (error) {
      console.error('Error fetching indicators:', error);
    }
  };

  // Fetch latest signal
  const fetchLatestSignal = async () => {
    try {
      const response = await axios.get('/api/latest-signal');
      if (response.data && !response.data.message) {
        setLastSignal(response.data);
      }
    } catch (error) {
      console.error('Error fetching latest signal:', error);
    }
  };

  // Show signal notification
  const showSignalNotification = (signal) => {
    setMessage(`🎯 ${signal.strategy_name} - ${signal.direction} (${signal.confidence ? (signal.confidence * 100).toFixed(0) : '0'}% confidence)`);
    setMessageType('info');
    
    // Clear message after 5 seconds
    setTimeout(() => {
      setMessage('');
    }, 5000);
  };

  // Update balance with notification
  const updateBalance = (newBalance) => {
    const oldBalance = balanceRef.current;
    setBalance(newBalance);
    balanceRef.current = newBalance;
    
    // Recalculate real profit when balance changes
    if (initialBalanceRef.current > 0) {
      const calculatedProfit = newBalance - initialBalanceRef.current;
      setRealProfit(calculatedProfit);
    }
    
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
  };

  // Update trade history with better handling of completed trades
  const updateTradeHistory = (newTrades) => {
    setTradeHistory(prev => {
      // If we receive a single trade update
      if (!Array.isArray(newTrades)) {
        const singleTrade = newTrades;
        const updated = [...prev];
        
        // Find and update existing trade or add new one
        const existingIndex = updated.findIndex(t => t.id === singleTrade.id);
        if (existingIndex >= 0) {
          // Update existing trade - preserve all data but ensure status update is applied
          updated[existingIndex] = { ...updated[existingIndex], ...singleTrade };
          
          // IMPORTANT FIX: Force explicit status logging for debugging
          if (singleTrade.status === 'completed' && updated[existingIndex].status === 'completed') {
            console.log(`✅ Trade ${singleTrade.id} marked COMPLETED with result: ${singleTrade.result}, P&L: ${singleTrade.profit_loss}`);
          }
          
          console.log(`🔄 Updated trade ${singleTrade.id} status: ${singleTrade.status}`);
        } else {
          // Add new trade at beginning
          updated.unshift(singleTrade);
          console.log(`➕ Added new trade ${singleTrade.id} status: ${singleTrade.status}`);
        }
        
        return updated;
      }
      
      // If we receive an array of trades (full refresh)
      if (Array.isArray(newTrades) && newTrades.length > 0) {
        // IMPORTANT FIX: Log trade statuses to help diagnose issues
        const activeCount = newTrades.filter(t => t.status === 'active').length;
        const completedCount = newTrades.filter(t => t.status === 'completed').length;
        
        console.log(`📋 Full trade history refresh: ${newTrades.length} trades (${activeCount} active, ${completedCount} completed)`);
        return newTrades;
      }
      
      return prev;
    });
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
      
      // Poll for connection status
      let attempts = 0;
      const maxAttempts = 15; // Increased attempts
      const checkStatus = setInterval(async () => {
        try {
          attempts++;
          const response = await axios.get('/api/connection-status');
          if (response.data.connected) {
            clearInterval(checkStatus);
            setIsConnected(true);
            setBalance(response.data.balance || 0);
            balanceRef.current = response.data.balance || 0;
            
            // Properly set initial balance
            if (response.data.initial_balance && response.data.initial_balance > 0) {
              setInitialBalance(response.data.initial_balance);
              initialBalanceRef.current = response.data.initial_balance;
              // Also calculate real profit right away
              const calculatedProfit = (response.data.balance || 0) - response.data.initial_balance;
              setRealProfit(calculatedProfit);
            }
            setMessage('Successfully connected to Deriv API');
            setMessageType('success');
            
            // Start heartbeat immediately
            startHeartbeat();
          } else if (attempts >= maxAttempts) {
            clearInterval(checkStatus);
            setMessage('Connection timed out. Please try again.');
            setMessageType('error');
          }
        } catch (error) {
          console.error('Error checking connection:', error);
          if (attempts >= maxAttempts) {
            clearInterval(checkStatus);
            setMessage('Connection failed. Please try again.');
            setMessageType('error');
          }
        }
      }, 1000);
      
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to connect');
      setMessageType('error');
    }
  };

  const handleStartTrading = async () => {
    try {
      await axios.post('/api/start-trading', {
        amount: tradeAmount
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
      stopPolling();
      stopHeartbeat();
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to disconnect');
      setMessageType('error');
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
      }, 3000);
    } catch (error) {
      setMessage(error.response?.data?.error || 'Failed to refresh balance');
      setMessageType('error');
    }
  };

  // Add a manual refresh button function
  const handleRefreshTrades = async () => {
    setMessage('Refreshing trade history...');
    setMessageType('info');
    
    try {
      await fetchTradeHistory();
      setMessage('Trade history refreshed successfully');
      setMessageType('success');
      
      // Clear message after 3 seconds
      setTimeout(() => {
        setMessage('');
      }, 3000);
    } catch (error) {
      setMessage('Failed to refresh trade history');
      setMessageType('error');
    }
  };

  const formatCurrency = (amount) => {
    if (amount === undefined || amount === null) {
      amount = 0;
    }
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
        <p>Automated 1-Tick trading for Volatility 100 Index</p>
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
                  onChange={(e) => setTradeAmount(parseFloat(e.target.value) || 1)}
                  min="1"
                  step="0.1"
                  disabled={isTrading}
                />
              </div>

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
              <div className="stat-value">{stats.total_trades || 0}</div>
              <div className="stat-label">Total Trades</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{stats.winning_rate !== undefined ? safeToFixed(stats.winning_rate) : '0.0'}%</div>
              <div className="stat-label">Win Rate</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{stats.winning_trades || 0}</div>
              <div className="stat-label">Wins</div>
            </div>
            
            <div className="stat-item">
              <div className="stat-value">{stats.losing_trades || 0}</div>
              <div className="stat-label">Losses</div>
            </div>
          </div>

          {/* Updated Starting Balance and Real Profit information */}
          <div className="balance-section">
            <div className="balance-row">
              <span>Starting Balance:</span>
              <span>{formatCurrency(initialBalance || 0)}</span>
            </div>
            <div className="balance-row">
              <span>Current Balance:</span>
              <span>{formatCurrency(balance || 0)}</span>
            </div>
            <div className="balance-row highlight">
              <span>Real Profit:</span>
              <span className={realProfit >= 0 ? 'profit' : 'loss'}>
                {realProfit !== 0 ? formatCurrency(realProfit) : '$0.00'}
              </span>
            </div>
          </div>

          <div className="balance-display" style={{ fontSize: '1.5rem', marginTop: '20px' }}>
            Total P&L: <span className={realProfit >= 0 ? 'profit' : 'loss'}>
              {realProfit !== 0 ? formatCurrency(realProfit) : '$0.00'}
            </span>
          </div>
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
                    <div>Confidence: {lastSignal.confidence !== undefined ? (lastSignal.confidence * 100).toFixed(0) : '0'}%</div>
                    <div>Hold Time: {lastSignal.hold_time}s</div>
                  </div>
                  <div className="signal-reason">{lastSignal.entry_reason}</div>
                  <div className="signal-conditions">
                    <strong>Conditions Met:</strong>
                    <ul>
                      {(lastSignal.conditions_met || []).map((condition, index) => (
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
                  <div className={`indicator-value ${(indicators.rsi || 0) > 70 ? 'overbought' : (indicators.rsi || 0) < 30 ? 'oversold' : 'neutral'}`}>
                    {indicators.rsi !== undefined ? safeToFixed(indicators.rsi) : '0.0'}
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">MACD</div>
                  <div className={`indicator-value ${(indicators.macd || 0) > 0 ? 'bullish' : 'bearish'}`}>
                    {indicators.macd !== undefined ? safeToFixed(indicators.macd, 4) : '0.0000'}
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">Momentum</div>
                  <div className={`indicator-value ${(indicators.momentum || 0) > 0 ? 'bullish' : 'bearish'}`}>
                    {indicators.momentum !== undefined ? safeToFixed(indicators.momentum, 2) : '0.00'}%
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">Volatility</div>
                  <div className={`indicator-value ${(indicators.volatility || 0) > 2 ? 'high' : (indicators.volatility || 0) < 0.5 ? 'low' : 'normal'}`}>
                    {indicators.volatility !== undefined ? safeToFixed(indicators.volatility, 2) : '0.00'}%
                  </div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">BB Upper</div>
                  <div className="indicator-value">{indicators.bb_upper !== undefined ? safeToFixed(indicators.bb_upper, 5) : '0.00000'}</div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">BB Lower</div>
                  <div className="indicator-value">{indicators.bb_lower !== undefined ? safeToFixed(indicators.bb_lower, 5) : '0.00000'}</div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">EMA5</div>
                  <div className="indicator-value">{indicators.ema5 !== undefined ? safeToFixed(indicators.ema5, 5) : '0.00000'}</div>
                </div>
                
                <div className="indicator-item">
                  <div className="indicator-label">Tick Count</div>
                  <div className="indicator-value">{indicators.tick_count || 0}</div>
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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
          <h2>Trade History</h2>
          <button 
            className="btn btn-secondary btn-small"
            onClick={handleRefreshTrades}
            style={{ fontSize: '0.8rem', padding: '5px 10px' }}
            title="Refresh trade history"
          >
            🔄 Refresh
          </button>
        </div>
        
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
                      {trade.strategy_confidence !== undefined && (
                        <div className="confidence-badge">
                          {(trade.strategy_confidence * 100).toFixed(0)}% confidence
                        </div>
                      )}
                    </div>
                  )}
                  {trade.entry_reason && (
                    <div className="entry-reason">{trade.entry_reason}</div>
                  )}
                  {/* Debug info for trade status */}
                  {trade.completed_at && (
                    <div className="completion-info" style={{ fontSize: '0.8rem', color: '#666' }}>
                      Completed: {formatTime(trade.completed_at)}
                    </div>
                  )}
                </div>
                
                <div className="trade-result">
                  <div className={`trade-status status-${trade.status || 'unknown'}`}>
                    {trade.status === 'active' ? '⏳ ACTIVE' : 
                     trade.status === 'completed' ? 
                       (trade.result === 'win' ? '✅ WIN' : '❌ LOSS') : 
                       '❓ UNKNOWN'}
                  </div>
                  {trade.profit_loss !== undefined && trade.status === 'completed' && (
                    <div className={`profit-loss ${trade.profit_loss >= 0 ? 'profit' : 'loss'}`}>
                      {trade.profit_loss >= 0 ? '+' : ''}{formatCurrency(trade.profit_loss)}
                    </div>
                  )}
                  {trade.actual_win_probability !== undefined && (
                    <div className="win-probability">
                      {(trade.actual_win_probability * 100).toFixed(0)}% win chance
                    </div>
                  )}
                  {/* Show outcome source for debugging */}
                  {trade.outcome_source && (
                    <div className="outcome-source" style={{ fontSize: '0.7rem', color: '#888' }}>
                      {trade.outcome_source === 'deriv_api' ? '🔗 Real' : '🔄 Sim'}
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
