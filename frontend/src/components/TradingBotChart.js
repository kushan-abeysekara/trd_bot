import React, { useRef, useEffect, useState } from 'react';
import { tradingAPI } from '../services/api';
import { LineChart, Activity, TrendingUp, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

const TradingBotChart = ({ onPriceUpdate, selectedStrategy }) => {
  const [chartData, setChartData] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [updateInterval, setUpdateInterval] = useState(500); // milliseconds
  const [indicators, setIndicators] = useState({
    rsi: 50,
    macd: 0,
    macdSignal: 0,
    momentum: 0,
    volatility: 0,
    bollingerUpper: 0,
    bollingerMiddle: 0,
    bollingerLower: 0
  });
  const [isRealtime, setIsRealtime] = useState(true);
  const [tradeSignals, setTradeSignals] = useState([]);
  const [timeframe, setTimeframe] = useState('1m');
  const [isTimeframeOpen, setIsTimeframeOpen] = useState(false);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const updateTimerRef = useRef(null);

  const timeframeOptions = [
    { value: '10s', label: '10 Seconds' },
    { value: '30s', label: '30 Seconds' },
    { value: '1m', label: '1 Minute' },
    { value: '5m', label: '5 Minutes' },
    { value: '15m', label: '15 Minutes' },
  ];

  // Fetch initial chart data
  useEffect(() => {
    fetchChartData();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (updateTimerRef.current) {
        clearInterval(updateTimerRef.current);
      }
    };
  }, [timeframe]);

  // Connect to WebSocket for real-time data
  useEffect(() => {
    if (isRealtime) {
      connectWebSocket();
      startRealtimeUpdates();
    } else {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (updateTimerRef.current) {
        clearInterval(updateTimerRef.current);
      }
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (updateTimerRef.current) {
        clearInterval(updateTimerRef.current);
      }
    };
  }, [isRealtime]);

  // Update chart when data changes
  useEffect(() => {
    if (chartData.length > 0) {
      drawChart();
    }
  }, [chartData, indicators, tradeSignals]);

  // Fetch technical indicators
  useEffect(() => {
    const fetchIndicators = async () => {
      try {
        const response = await tradingAPI.getBotStatus();
        if (response.data && response.data.success) {
          const { rsi, macd, bollinger_bands, momentum, volatility } = response.data.technical_indicators || {};
          
          setIndicators({
            rsi: rsi || 50,
            macd: macd?.macd || 0,
            macdSignal: macd?.signal || 0,
            momentum: momentum || 0,
            volatility: volatility || 0,
            bollingerUpper: bollinger_bands?.upper || 0,
            bollingerMiddle: bollinger_bands?.middle || 0,
            bollingerLower: bollinger_bands?.lower || 0
          });
        }
      } catch (error) {
        console.error('Failed to fetch indicators:', error);
      }
    };

    fetchIndicators();
    const indicatorsInterval = setInterval(fetchIndicators, 2000);

    return () => clearInterval(indicatorsInterval);
  }, []);

  const fetchChartData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await tradingAPI.getMarketData(timeframe);
      
      if (response.data && response.data.success) {
        const data = response.data.prices || [];
        setChartData(data);
        
        if (data.length > 0) {
          const latestPrice = data[data.length - 1].price;
          setCurrentPrice(latestPrice);
          if (onPriceUpdate) {
            onPriceUpdate(latestPrice);
          }
        }
      } else {
        // Use mock data if API fails or returns empty
        generateMockData();
      }
    } catch (error) {
      console.error('Failed to fetch chart data:', error);
      setError('Failed to load chart data');
      generateMockData();
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockData = () => {
    const mockData = [];
    let basePrice = 500 + Math.random() * 50;
    const now = Date.now();
    const interval = timeframe === '1m' ? 60000 : 
                    timeframe === '5m' ? 300000 : 
                    timeframe === '15m' ? 900000 : 
                    timeframe === '30s' ? 30000 : 10000; // Default to 10s
    
    const dataPoints = 100;
    
    for (let i = dataPoints - 1; i >= 0; i--) {
      const timestamp = now - (i * interval);
      const volatility = 0.005; // 0.5% volatility
      
      const change = (Math.random() - 0.5) * basePrice * volatility;
      basePrice = Math.max(basePrice + change, 100);
      
      mockData.push({
        timestamp,
        price: basePrice,
        time: new Date(timestamp).toLocaleTimeString(),
        volume: Math.floor(Math.random() * 100) + 50
      });
    }
    
    setChartData(mockData);
    
    if (mockData.length > 0) {
      const latestPrice = mockData[mockData.length - 1].price;
      setCurrentPrice(latestPrice);
      if (onPriceUpdate) {
        onPriceUpdate(latestPrice);
      }
    }
  };

  const connectWebSocket = () => {
    // Placeholder for actual WebSocket implementation
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Mock WebSocket behavior
    console.log('WebSocket connected for real-time chart updates');
  };

  const startRealtimeUpdates = () => {
    if (updateTimerRef.current) {
      clearInterval(updateTimerRef.current);
    }

    updateTimerRef.current = setInterval(() => {
      if (chartData.length > 0) {
        const lastPoint = chartData[chartData.length - 1];
        const volatility = 0.002; // 0.2% price movement
        const newPrice = lastPoint.price * (1 + (Math.random() - 0.5) * volatility);
        const newTimestamp = Date.now();
        
        const newPoint = {
          timestamp: newTimestamp,
          price: newPrice,
          time: new Date(newTimestamp).toLocaleTimeString(),
          volume: Math.floor(Math.random() * 100) + 50
        };

        // Generate random trade signals occasionally
        if (Math.random() < 0.05) { // 5% chance of a signal
          const signal = {
            timestamp: newTimestamp,
            price: newPrice,
            direction: Math.random() > 0.5 ? 'RISE' : 'FALL',
            strategy: selectedStrategy?.name || 'Unknown Strategy',
            confidence: Math.floor(Math.random() * 30) + 60 // 60-90% confidence
          };
          
          setTradeSignals(prev => [...prev.slice(-9), signal]); // Keep last 10 signals
        }

        // Add new point and keep the last 100 points
        setChartData(prev => [...prev.slice(1), newPoint]);
        setCurrentPrice(newPrice);
        
        if (onPriceUpdate) {
          onPriceUpdate(newPrice);
        }

        // Update indicators
        setIndicators(prev => ({
          ...prev,
          rsi: Math.min(Math.max(prev.rsi + (Math.random() - 0.5) * 5, 10), 90),
          macd: prev.macd + (Math.random() - 0.5) * 0.1,
          macdSignal: prev.macdSignal + (Math.random() - 0.5) * 0.05,
          momentum: prev.momentum + (Math.random() - 0.5) * 0.02,
          volatility: Math.max(prev.volatility + (Math.random() - 0.5) * 0.001, 0.001)
        }));
      }
    }, updateInterval);
  };

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas || chartData.length === 0) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate price range with margin
    const prices = chartData.map(d => d.price);
    let minPrice = Math.min(...prices);
    let maxPrice = Math.max(...prices);
    
    // Add 5% margin to price range
    const priceRange = maxPrice - minPrice;
    minPrice = Math.max(minPrice - priceRange * 0.05, 0);
    maxPrice = maxPrice + priceRange * 0.05;
    
    // Draw grid
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 0.5;
    
    // Horizontal grid lines and price labels
    for (let i = 0; i <= 5; i++) {
      const y = (height - 60) * (1 - i / 5) + 20;
      const price = minPrice + (maxPrice - minPrice) * i / 5;
      
      ctx.beginPath();
      ctx.moveTo(60, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();
      
      // Price labels
      ctx.fillStyle = '#6B7280';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(price.toFixed(2), 55, y + 4);
    }
    
    // Vertical grid lines (time)
    const timeLabelsCount = 5;
    for (let i = 0; i <= timeLabelsCount; i++) {
      const x = (width - 80) * i / timeLabelsCount + 60;
      
      ctx.beginPath();
      ctx.moveTo(x, 20);
      ctx.lineTo(x, height - 40);
      ctx.stroke();
      
      // Time labels
      if (chartData.length > 0) {
        const dataIndex = Math.floor((chartData.length - 1) * i / timeLabelsCount);
        const point = chartData[dataIndex];
        const timeLabel = new Date(point.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        ctx.fillStyle = '#6B7280';
        ctx.font = '10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(timeLabel, x, height - 25);
      }
    }
    
    // Draw Bollinger Bands if available
    if (indicators.bollingerUpper > 0 && indicators.bollingerLower > 0) {
      // Upper band
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)'; // Red with transparency
      ctx.beginPath();
      
      chartData.forEach((point, index) => {
        const x = ((width - 80) * index / (chartData.length - 1)) + 60;
        const upperPrice = point.price * (1 + indicators.volatility * 2);
        const y = height - 40 - ((upperPrice - minPrice) / (maxPrice - minPrice)) * (height - 60);
        
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      
      ctx.stroke();
      
      // Lower band
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)'; // Blue with transparency
      ctx.beginPath();
      
      chartData.forEach((point, index) => {
        const x = ((width - 80) * index / (chartData.length - 1)) + 60;
        const lowerPrice = point.price * (1 - indicators.volatility * 2);
        const y = height - 40 - ((lowerPrice - minPrice) / (maxPrice - minPrice)) * (height - 60);
        
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      
      ctx.stroke();
    }
    
    // Draw price line
    ctx.strokeStyle = '#3B82F6'; // Blue
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    chartData.forEach((point, index) => {
      const x = ((width - 80) * index / (chartData.length - 1)) + 60;
      const y = height - 40 - ((point.price - minPrice) / (maxPrice - minPrice)) * (height - 60);
      
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    
    ctx.stroke();
    
    // Draw trade signals
    tradeSignals.forEach(signal => {
      const signalIndex = chartData.findIndex(point => point.timestamp >= signal.timestamp);
      if (signalIndex >= 0) {
        const x = ((width - 80) * signalIndex / (chartData.length - 1)) + 60;
        const y = height - 40 - ((signal.price - minPrice) / (maxPrice - minPrice)) * (height - 60);
        
        ctx.fillStyle = signal.direction === 'RISE' ? '#10B981' : '#EF4444'; // Green for rise, red for fall
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
    
    // Draw current price indicator
    if (currentPrice) {
      const lastPoint = chartData[chartData.length - 1];
      const x = width - 20;
      const y = height - 40 - ((currentPrice - minPrice) / (maxPrice - minPrice)) * (height - 60);
      
      ctx.fillStyle = currentPrice >= lastPoint.price ? '#10B981' : '#EF4444';
      ctx.beginPath();
      ctx.arc(x - 10, y, 5, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = currentPrice >= lastPoint.price ? '#10B981' : '#EF4444';
      ctx.textAlign = 'right';
      ctx.fillText(currentPrice.toFixed(2), x, y - 10);
    }
  };

  const toggleRealtimeUpdates = () => {
    setIsRealtime(!isRealtime);
  };

  const handleRefresh = () => {
    fetchChartData();
  };

  const handleTimeframeChange = (newTimeframe) => {
    setTimeframe(newTimeframe);
    setIsTimeframeOpen(false);
  };

  const getSelectedTimeframeLabel = () => {
    const selected = timeframeOptions.find(option => option.value === timeframe);
    return selected ? selected.label : '1 Minute';
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {/* Header with controls */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <LineChart className="w-6 h-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Real-time Trading Chart</h3>
        </div>
        
        <div className="flex items-center space-x-3">
          {/* Timeframe selector */}
          <div className="relative">
            <button
              onClick={() => setIsTimeframeOpen(!isTimeframeOpen)}
              className="flex items-center space-x-2 px-3 py-1.5 border border-gray-300 rounded-lg hover:border-gray-400 focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
            >
              <span className="text-gray-700">{getSelectedTimeframeLabel()}</span>
              {isTimeframeOpen ? (
                <ChevronUp className="w-4 h-4 text-gray-500" />
              ) : (
                <ChevronDown className="w-4 h-4 text-gray-500" />
              )}
            </button>
            
            {isTimeframeOpen && (
              <div className="absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-lg shadow-lg z-10">
                {timeframeOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => handleTimeframeChange(option.value)}
                    className={`w-full text-left px-4 py-2 hover:bg-gray-50 ${timeframe === option.value ? 'text-blue-600 bg-blue-50' : 'text-gray-700'}`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}
          </div>
          
          {/* Realtime toggle */}
          <button
            onClick={toggleRealtimeUpdates}
            className={`flex items-center space-x-1 px-3 py-1.5 rounded-lg transition-colors ${isRealtime ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}
          >
            <Activity className="w-4 h-4" />
            <span className="text-sm">{isRealtime ? 'Live' : 'Paused'}</span>
          </button>
          
          {/* Refresh button */}
          <button
            onClick={handleRefresh}
            className="p-1.5 rounded-lg text-gray-600 hover:bg-gray-100 transition-colors"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>
      
      {/* Current price and indicators */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <div className="flex flex-col p-3 bg-gray-50 rounded-lg">
          <span className="text-sm text-gray-600">Current Price</span>
          <span className="text-lg font-bold text-gray-900">${currentPrice ? currentPrice.toFixed(2) : '---'}</span>
        </div>
        <div className="flex flex-col p-3 bg-gray-50 rounded-lg">
          <span className="text-sm text-gray-600">RSI</span>
          <span className={`text-lg font-bold ${
            indicators.rsi > 70 ? 'text-red-600' : 
            indicators.rsi < 30 ? 'text-green-600' : 'text-gray-900'
          }`}>
            {indicators.rsi.toFixed(1)}
          </span>
        </div>
        <div className="flex flex-col p-3 bg-gray-50 rounded-lg">
          <span className="text-sm text-gray-600">MACD</span>
          <span className={`text-lg font-bold ${indicators.macd > 0 ? 'text-green-600' : 'text-red-600'}`}>
            {indicators.macd.toFixed(3)}
          </span>
        </div>
        <div className="flex flex-col p-3 bg-gray-50 rounded-lg">
          <span className="text-sm text-gray-600">Volatility</span>
          <span className="text-lg font-bold text-gray-900">{(indicators.volatility * 100).toFixed(2)}%</span>
        </div>
      </div>
      
      {/* Chart */}
      <div className="relative mb-3" style={{ height: '300px' }}>
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex items-center justify-center bg-red-50 rounded-lg">
            <div className="text-red-600">
              <p className="font-medium">{error}</p>
              <button 
                onClick={handleRefresh}
                className="mt-2 px-3 py-1 bg-red-100 text-red-700 rounded-md hover:bg-red-200 text-sm"
              >
                Retry
              </button>
            </div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={800}
            height={300}
            className="w-full h-full border border-gray-200 rounded-lg"
          />
        )}
      </div>
      
      {/* Chart legend and info */}
      <div className="flex justify-between text-xs text-gray-500">
        <div className="flex items-center space-x-4">
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-blue-500 mr-1"></div>
            <span>Price</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-red-300 mr-1"></div>
            <span>Upper Band</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-blue-300 mr-1"></div>
            <span>Lower Band</span>
          </div>
        </div>
        <div className="flex items-center">
          <div className="flex items-center mr-3">
            <div className="w-2 h-2 rounded-full bg-green-500 mr-1"></div>
            <span>Buy Signal</span>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 rounded-full bg-red-500 mr-1"></div>
            <span>Sell Signal</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingBotChart;
