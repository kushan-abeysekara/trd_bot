import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Zap, Activity, TrendingUp, TrendingDown, RefreshCw, BarChart2,
  AlertCircle, Clock, Eye, EyeOff 
} from 'lucide-react';
import { tradingAPI } from '../services/api';

const TradingChart = ({ symbol, onTickData, height = 400 }) => {
  const [chartData, setChartData] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [priceChange, setPriceChange] = useState({ value: 0, isUp: false });
  const [lastDigit, setLastDigit] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [indicators, setIndicators] = useState({
    rsi: 50,
    macd: 0,
    signal: 0,
    momentum: 0,
    volatility: 0,
    bollingerUpper: 0,
    bollingerMiddle: 0,
    bollingerLower: 0
  });
  const [showIndicators, setShowIndicators] = useState(true);
  const [wsConnection, setWsConnection] = useState(null);
  const canvasRef = useRef(null);
  // Chart reference for future implementation
  // For future implementation of candlestick chart
  const [timeframe, setTimeframe] = useState('1m');
  const animationFrameRef = useRef(null);
  const [isChartAnimating] = useState(true);

  // Timeframe options
  const timeframes = [
    { value: '10s', label: '10s' },
    { value: '1m', label: '1m' },
    { value: '5m', label: '5m' },
    { value: '15m', label: '15m' },
    { value: '1h', label: '1h' }
  ];

  const fetchChartData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await tradingAPI.getVolatilityData(symbol || 'volatility_10_1s', timeframe, 100);
      
      if (response.data && response.data.success) {
        const data = response.data.prices || [];
        setChartData(data);
        
        if (data.length > 0) {
          const latestPrice = data[data.length - 1];
          const previousPrice = data.length > 1 ? data[data.length - 2].price : latestPrice.price;
          
          setCurrentPrice(latestPrice.price);
          updateLastDigit(latestPrice.price);
          setPriceChange({
            value: parseFloat((latestPrice.price - previousPrice).toFixed(5)),
            isUp: latestPrice.price >= previousPrice
          });
        }
        
        // Candlestick data will be implemented in future versions
      } else {
        setError('Failed to fetch market data');
      }
    } catch (err) {
      console.error('Error fetching chart data:', err);
      setError('Unable to load market data');
    } finally {
      setIsLoading(false);
    }
  }, [symbol, timeframe]);

  const fetchTechnicalIndicators = useCallback(async () => {
    try {
      const response = await tradingAPI.getTechnicalIndicators(symbol || 'volatility_10_1s');
      
      if (response.data && response.data.success) {
        setIndicators(response.data.indicators);
      }
    } catch (err) {
      console.error('Error fetching technical indicators:', err);
    }
  }, [symbol]);

  // Initialize websocket connection
  const initWebSocket = useCallback(() => {
    // Close existing connection if any
    if (wsConnection) {
      wsConnection.close();
    }
    
    // Get the WebSocket URL from environment or use default
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:5000/ws';
    const ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      // Subscribe to symbol
      ws.send(JSON.stringify({ 
        type: 'subscribe', 
        symbol: symbol || 'volatility_10_1s',
        timeframe: timeframe
      }));
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Handle different message types
      if (data.type === 'tick') {
        // Update price
        setCurrentPrice(data.price);
        updateLastDigit(data.price);
        
        // Calculate price change
        const prevPrice = chartData.length > 0 ? chartData[chartData.length - 1].price : data.price;
        setPriceChange({
          value: parseFloat((data.price - prevPrice).toFixed(5)),
          isUp: data.price >= prevPrice
        });
        
        // Update chart data
        setChartData(prevData => {
          const newData = [...prevData, { price: data.price, timestamp: data.timestamp }];
          // Keep only the last 100 data points for performance
          return newData.slice(-100);
        });
        
        // Notify parent component
        if (onTickData) {
          onTickData({
            price: data.price,
            lastDigit: String(data.price).slice(-1),
            timestamp: data.timestamp
          });
        }
      } else if (data.type === 'indicators') {
        // Update technical indicators
        setIndicators(data.indicators);
      } else if (data.type === 'candle') {
        // Candlestick data will be implemented in future versions
        console.log('Received candle data:', data.candle);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection error');
    };
    
    ws.onclose = (event) => {
      console.log('WebSocket connection closed', event);
      // Auto reconnect after 3 seconds
      setTimeout(() => {
        if (wsConnection === ws) {
          initWebSocket();
        }
      }, 3000);
    };
    
    setWsConnection(ws);
    
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [symbol, timeframe, chartData, wsConnection, onTickData]);

  // Initialize chart
  useEffect(() => {
    fetchChartData();
    fetchTechnicalIndicators();
    
    // Setup WebSocket connection
    initWebSocket();
    
    // Cleanup function
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [fetchChartData, fetchTechnicalIndicators, initWebSocket, wsConnection]);

  // Draw chart function
  useEffect(() => {
    if (!canvasRef.current || chartData.length === 0 || !isChartAnimating) return;
    
    const drawChart = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      
      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Set dimensions
      const width = canvas.width;
      const height = canvas.height;
      const padding = 40;
      const chartWidth = width - padding * 2;
      const chartHeight = height - padding * 2;
      
      // Get price range
      const prices = chartData.map(d => d.price);
      const minPrice = Math.min(...prices) * 0.9999;
      const maxPrice = Math.max(...prices) * 1.0001;
      const priceRange = maxPrice - minPrice;
      
      // Draw axes
      ctx.beginPath();
      ctx.strokeStyle = '#e2e8f0';
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, height - padding);
      ctx.lineTo(width - padding, height - padding);
      ctx.stroke();
      
      // Draw price scale
      ctx.fillStyle = '#64748b';
      ctx.font = '10px sans-serif';
      
      const priceStep = priceRange / 5;
      for (let i = 0; i <= 5; i++) {
        const price = minPrice + priceStep * i;
        const y = height - padding - (price - minPrice) / priceRange * chartHeight;
        
        // Grid line
        ctx.beginPath();
        ctx.strokeStyle = '#e2e8f0';
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
        
        // Price label
        ctx.fillText(price.toFixed(5), 5, y + 3);
      }
      
      // Draw time scale
      const timeStep = chartData.length / 5;
      for (let i = 0; i < 5; i++) {
        const dataIndex = Math.floor(i * timeStep);
        if (dataIndex < chartData.length) {
          const x = padding + (dataIndex / (chartData.length - 1)) * chartWidth;
          const time = new Date(chartData[dataIndex].timestamp);
          const timeStr = `${time.getHours()}:${time.getMinutes().toString().padStart(2, '0')}`;
          
          // Time label
          ctx.fillText(timeStr, x - 10, height - padding + 15);
        }
      }
      
      // Draw price line
      ctx.beginPath();
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      
      for (let i = 0; i < chartData.length; i++) {
        const x = padding + (i / (chartData.length - 1)) * chartWidth;
        const y = height - padding - (chartData[i].price - minPrice) / priceRange * chartHeight;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.stroke();
      
      // Draw current price indicator
      if (chartData.length > 0) {
        const lastPrice = chartData[chartData.length - 1].price;
        const y = height - padding - (lastPrice - minPrice) / priceRange * chartHeight;
        
        // Price line
        ctx.beginPath();
        ctx.strokeStyle = '#94a3b8';
        ctx.setLineDash([2, 2]);
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Price box
        ctx.fillStyle = priceChange.isUp ? '#10b981' : '#ef4444';
        ctx.fillRect(width - padding + 1, y - 10, 40, 20);
        
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px sans-serif';
        ctx.fillText(lastPrice.toFixed(5), width - padding + 5, y + 4);
      }
      
      // Draw Bollinger Bands if available and indicators are shown
      if (showIndicators && indicators.bollingerUpper && indicators.bollingerLower) {
        // Calculate positions
        const upperY = height - padding - (indicators.bollingerUpper - minPrice) / priceRange * chartHeight;
        const middleY = height - padding - (indicators.bollingerMiddle - minPrice) / priceRange * chartHeight;
        const lowerY = height - padding - (indicators.bollingerLower - minPrice) / priceRange * chartHeight;
        
        // Draw upper band
        ctx.beginPath();
        ctx.strokeStyle = '#9333ea';
        ctx.setLineDash([2, 2]);
        ctx.moveTo(padding, upperY);
        ctx.lineTo(width - padding, upperY);
        ctx.stroke();
        
        // Draw middle band
        ctx.beginPath();
        ctx.strokeStyle = '#8b5cf6';
        ctx.setLineDash([2, 2]);
        ctx.moveTo(padding, middleY);
        ctx.lineTo(width - padding, middleY);
        ctx.stroke();
        
        // Draw lower band
        ctx.beginPath();
        ctx.strokeStyle = '#9333ea';
        ctx.setLineDash([2, 2]);
        ctx.moveTo(padding, lowerY);
        ctx.lineTo(width - padding, lowerY);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // Request next animation frame
      if (isChartAnimating) {
        animationFrameRef.current = requestAnimationFrame(drawChart);
      }
    };
    
    // Start animation
    drawChart();
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [chartData, isChartAnimating, indicators, showIndicators, priceChange.isUp]);

  // Helper function to update last digit
  const updateLastDigit = (price) => {
    const priceStr = price.toString();
    const digit = priceStr.slice(-1);
    setLastDigit(digit);
  };

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
    // Reconnect WebSocket with new timeframe
    if (wsConnection) {
      wsConnection.send(JSON.stringify({ 
        type: 'change_timeframe', 
        timeframe: newTimeframe 
      }));
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-100 overflow-hidden">
      {/* Chart Header */}
      <div className="border-b border-gray-100 px-4 py-3 flex items-center justify-between">
        <div>
          <h2 className="font-semibold text-gray-800">
            {symbol || 'Volatility 10 (1s) Index'}
          </h2>
          <div className="flex items-center mt-1">
            <span className="text-lg font-semibold font-mono">{currentPrice ? currentPrice.toFixed(5) : '0.00000'}</span>
            <div className={`ml-2 ${priceChange.isUp ? 'text-green-600' : 'text-red-600'} flex items-center text-xs font-semibold`}>
              {priceChange.isUp ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
              {priceChange.value > 0 ? '+' : ''}{priceChange.value.toFixed(5)}
            </div>
          </div>
        </div>

        {/* Control buttons */}
        <div className="flex items-center space-x-2">
          {/* Timeframe selector */}
          <div className="flex border border-gray-200 rounded-md">
            {timeframes.map(tf => (
              <button
                key={tf.value}
                className={`px-2 py-1 text-xs font-medium ${timeframe === tf.value 
                  ? 'bg-blue-100 text-blue-700 border-blue-400' 
                  : 'text-gray-600 hover:bg-gray-100'}`}
                onClick={() => handleTimeframeChange(tf.value)}
              >
                {tf.label}
              </button>
            ))}
          </div>

          {/* Toggle indicators */}
          <button 
            onClick={() => setShowIndicators(!showIndicators)}
            className={`p-1.5 rounded-md ${showIndicators ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
            title={showIndicators ? 'Hide indicators' : 'Show indicators'}
          >
            {showIndicators ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>

          {/* Refresh button */}
          <button 
            onClick={fetchChartData}
            className="p-1.5 rounded-md text-gray-600 hover:bg-gray-100"
            title="Refresh chart data"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Loading and error states */}
      {isLoading && chartData.length === 0 && (
        <div className="flex items-center justify-center" style={{ height: `${height}px` }}>
          <div className="flex flex-col items-center text-gray-500">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-2"></div>
            <p>Loading chart data...</p>
          </div>
        </div>
      )}

      {error && chartData.length === 0 && (
        <div className="flex items-center justify-center" style={{ height: `${height}px` }}>
          <div className="flex flex-col items-center text-red-500">
            <AlertCircle className="h-8 w-8 mb-2" />
            <p>{error}</p>
            <button 
              onClick={fetchChartData}
              className="mt-2 px-3 py-1 bg-blue-100 text-blue-700 rounded-md text-sm hover:bg-blue-200 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      {/* Chart canvas */}
      {chartData.length > 0 && (
        <div className="relative" style={{ height: `${height}px` }}>
          <canvas 
            ref={canvasRef} 
            width="800" 
            height={height} 
            className="w-full h-full" 
          />

          {/* Last digit indicator */}
          {lastDigit !== null && (
            <div className="absolute top-4 right-4 w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg text-white"
                style={{ 
                  backgroundColor: parseInt(lastDigit) % 2 === 0 ? '#ef4444' : '#10b981',
                  boxShadow: '0 0 10px rgba(0,0,0,0.2)'
                }}>
              {lastDigit}
            </div>
          )}
        </div>
      )}

      {/* Technical indicators panel */}
      {showIndicators && (
        <div className="border-t border-gray-100 px-4 py-3">
          <div className="flex flex-wrap -mx-2">
            {/* RSI */}
            <div className="px-2 w-1/2 md:w-1/4 mb-2">
              <div className="flex items-center">
                <Activity className="w-4 h-4 text-purple-600 mr-1" />
                <span className="text-xs font-medium text-gray-600">RSI:</span>
                <span className={`ml-1 text-sm font-semibold ${
                  indicators.rsi > 70 ? 'text-red-600' : 
                  indicators.rsi < 30 ? 'text-green-600' : 'text-gray-800'
                }`}>
                  {indicators.rsi.toFixed(1)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                <div 
                  className="bg-purple-600 h-1.5 rounded-full" 
                  style={{ width: `${Math.min(100, Math.max(0, indicators.rsi))}%` }}
                ></div>
              </div>
            </div>

            {/* MACD */}
            <div className="px-2 w-1/2 md:w-1/4 mb-2">
              <div className="flex items-center">
                <BarChart2 className="w-4 h-4 text-blue-600 mr-1" />
                <span className="text-xs font-medium text-gray-600">MACD:</span>
                <span className={`ml-1 text-sm font-semibold ${
                  indicators.macd > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {indicators.macd.toFixed(3)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                <div 
                  className={`h-1.5 rounded-full ${indicators.macd > 0 ? 'bg-green-500' : 'bg-red-500'}`} 
                  style={{ 
                    width: `${Math.min(100, Math.max(0, 50 + Math.min(Math.abs(indicators.macd) * 100, 50) * (indicators.macd > 0 ? 1 : -1)))}%` 
                  }}
                ></div>
              </div>
            </div>

            {/* Momentum */}
            <div className="px-2 w-1/2 md:w-1/4 mb-2">
              <div className="flex items-center">
                <TrendingUp className="w-4 h-4 text-amber-600 mr-1" />
                <span className="text-xs font-medium text-gray-600">Momentum:</span>
                <span className={`ml-1 text-sm font-semibold ${
                  indicators.momentum > 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {indicators.momentum.toFixed(4)}
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                <div 
                  className={`h-1.5 rounded-full ${indicators.momentum > 0 ? 'bg-green-500' : 'bg-red-500'}`} 
                  style={{ 
                    width: `${Math.min(100, Math.max(0, 50 + Math.min(Math.abs(indicators.momentum) * 500, 50) * (indicators.momentum > 0 ? 1 : -1)))}%` 
                  }}
                ></div>
              </div>
            </div>

            {/* Volatility */}
            <div className="px-2 w-1/2 md:w-1/4 mb-2">
              <div className="flex items-center">
                <Zap className="w-4 h-4 text-orange-600 mr-1" />
                <span className="text-xs font-medium text-gray-600">Volatility:</span>
                <span className="ml-1 text-sm font-semibold text-gray-800">
                  {(indicators.volatility * 100).toFixed(2)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                <div 
                  className="bg-orange-500 h-1.5 rounded-full" 
                  style={{ width: `${Math.min(100, indicators.volatility * 3000)}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="mt-2 text-xs text-gray-500 flex items-center">
            <Clock className="w-3 h-3 mr-1" />
            <span>Last update: {new Date().toLocaleTimeString()}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingChart;
