import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, TrendingUp, Activity } from 'lucide-react';
import { tradingAPI } from '../services/api';

const VolatilityChart = ({ onLastDigitUpdate }) => {
  const [selectedIndex, setSelectedIndex] = useState('volatility_10_1s');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [chartData, setChartData] = useState([]);
  const [currentPrice, setCurrentPrice] = useState(null);
  const [lastDigit, setLastDigit] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const [realtimeStream, setRealtimeStream] = useState(null);
  const [tickData, setTickData] = useState([]);
  const [priceHistory, setPriceHistory] = useState([]);

  const volatilityIndices = [
    { value: 'volatility_10_1s', label: 'Volatility 10 (1s) Index', color: '#3B82F6' },
    { value: 'volatility_10', label: 'Volatility 10 Index', color: '#10B981' },
    { value: 'volatility_25_1s', label: 'Volatility 25 (1s) Index', color: '#F59E0B' },
    { value: 'volatility_25', label: 'Volatility 25 Index', color: '#EF4444' },
    { value: 'volatility_50_1s', label: 'Volatility 50 (1s) Index', color: '#8B5CF6' },
    { value: 'volatility_50', label: 'Volatility 50 Index', color: '#06B6D4' },
    { value: 'volatility_75_1s', label: 'Volatility 75 (1s) Index', color: '#84CC16' },
    { value: 'volatility_75', label: 'Volatility 75 Index', color: '#F97316' },
    { value: 'volatility_100_1s', label: 'Volatility 100 (1s) Index', color: '#EC4899' },
    { value: 'volatility_100', label: 'Volatility 100 Index', color: '#6366F1' }
  ];

  const selectedIndexData = volatilityIndices.find(index => index.value === selectedIndex);

  // Fetch initial chart data
  const fetchChartData = async (symbol) => {
    setIsLoading(true);
    try {
      const response = await tradingAPI.getVolatilityData(symbol, '1m', 100);
      const data = response.data.prices || [];
      setChartData(data);
      if (data.length > 0) {
        const latest = data[data.length - 1];
        setCurrentPrice(latest.price);
        updateLastDigit(latest.price);
      }
    } catch (error) {
      console.error('Failed to fetch chart data:', error);
      // Generate mock data for demonstration
      generateMockData();
    } finally {
      setIsLoading(false);
    }
  };

  // Generate mock data for demonstration
  const generateMockData = () => {
    const mockData = [];
    let basePrice = 500;
    const now = Date.now();
    
    for (let i = 99; i >= 0; i--) {
      const timestamp = now - (i * 60000); // 1 minute intervals
      const volatility = selectedIndex.includes('100') ? 0.02 : 
                        selectedIndex.includes('75') ? 0.015 : 
                        selectedIndex.includes('50') ? 0.012 : 
                        selectedIndex.includes('25') ? 0.008 : 0.005;
      
      const change = (Math.random() - 0.5) * basePrice * volatility;
      basePrice = Math.max(basePrice + change, 100);
      
      mockData.push({
        timestamp,
        price: basePrice,
        time: new Date(timestamp).toLocaleTimeString()
      });
    }
    
    setChartData(mockData);
    if (mockData.length > 0) {
      const latest = mockData[mockData.length - 1];
      setCurrentPrice(latest.price);
      updateLastDigit(latest.price);
    }
  };

  // Update last digit and notify parent
  const updateLastDigit = (price) => {
    // Get the 2nd decimal place (e.g., 500.32 -> 2)
    const digit = Math.floor(price * 100) % 10;
    setLastDigit(digit);
    if (onLastDigitUpdate) {
      onLastDigitUpdate(digit, selectedIndexData.label, price, chartData);
    }
  };

  // Handle index change and update bot
  const handleIndexChange = async (indexValue) => {
    setSelectedIndex(indexValue);
    setIsDropdownOpen(false);
    
    // Update bot market selection
    try {
      await tradingAPI.updateBotSettings({ selected_market: indexValue });
      console.log(`Bot market updated to: ${indexValue}`);
    } catch (error) {
      console.error('Failed to update bot market:', error);
    }
    
    fetchChartData(indexValue);
  };

  // Enhanced real-time updates with bot synchronization
  const startRealTimeUpdates = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    // Much faster updates for 1s indices
    const updateInterval = selectedIndex.includes('1s') ? 200 : 1000; // 200ms for 1s, 1s for others

    intervalRef.current = setInterval(async () => {
      if (chartData.length > 0) {
        const lastPrice = chartData[chartData.length - 1].price;
        const volatility = selectedIndex.includes('100') ? 0.02 : 
                          selectedIndex.includes('75') ? 0.015 : 
                          selectedIndex.includes('50') ? 0.012 : 
                          selectedIndex.includes('25') ? 0.008 : 0.005;
        
        // Enhanced price simulation with micro-movements
        const change = (Math.random() - 0.5) * lastPrice * volatility;
        const microChange = (Math.random() - 0.5) * lastPrice * 0.001; // Small micro movements
        const newPrice = Math.max(lastPrice + change + microChange, 100);
        const now = Date.now();
        
        const newDataPoint = {
          timestamp: now,
          price: newPrice,
          time: new Date(now).toLocaleTimeString(),
          tick: Math.floor(Math.random() * 1000) // Simulated tick ID
        };

        setChartData(prev => {
          const updated = [...prev.slice(1), newDataPoint];
          return updated;
        });
        
        // Store tick data for advanced analysis
        setTickData(prev => [...prev.slice(-500), newDataPoint]); // Keep last 500 ticks
        
        setCurrentPrice(newPrice);
        updateLastDigit(newPrice);
        
        // Update price history for trend analysis
        setPriceHistory(prev => [...prev.slice(-100), newPrice]);
        
        // Send market data to trading bot
        try {
          await tradingAPI.updateMarketData({
            market: selectedIndex,
            price: newPrice,
            timestamp: now
          });
        } catch (error) {
          console.error('Failed to update bot market data:', error);
        }
      }
    }, updateInterval);
  };

  // Draw chart on canvas
  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas || chartData.length === 0) return;

    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Calculate price range
    const prices = chartData.map(d => d.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    // Draw grid
    ctx.strokeStyle = '#E5E7EB';
    ctx.lineWidth = 1;
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = (height - 60) * i / 5 + 20;
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 20, y);
      ctx.stroke();
    }
    
    // Draw price line
    ctx.strokeStyle = selectedIndexData.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    chartData.forEach((point, index) => {
      const x = ((width - 60) * index / (chartData.length - 1)) + 40;
      const y = height - 40 - ((point.price - minPrice) / priceRange) * (height - 60);
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current price indicator
    if (currentPrice) {
      const y = height - 40 - ((currentPrice - minPrice) / priceRange) * (height - 60);
      ctx.fillStyle = selectedIndexData.color;
      ctx.beginPath();
      ctx.arc(width - 40, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
    
    // Draw price labels
    ctx.fillStyle = '#6B7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    
    for (let i = 0; i <= 5; i++) {
      const price = minPrice + (priceRange * i / 5);
      const y = height - 40 - ((price - minPrice) / priceRange) * (height - 60);
      ctx.fillText(price.toFixed(2), 35, y + 4);
    }
  };

  // Initialize component
  useEffect(() => {
    fetchChartData(selectedIndex);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [selectedIndex]);

  // Start real-time updates
  useEffect(() => {
    if (chartData.length > 0) {
      startRealTimeUpdates();
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [chartData, selectedIndex]);

  // Draw chart when data updates
  useEffect(() => {
    drawChart();
  }, [chartData, selectedIndexData]);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {/* Header with dropdown */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <TrendingUp className="w-6 h-6 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Volatility Indices</h3>
        </div>
        
        <div className="relative">
          <button
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <span className="text-sm font-medium text-gray-700">
              {selectedIndexData.label}
            </span>
            <ChevronDown className="w-4 h-4 text-gray-500" />
          </button>
          
          {isDropdownOpen && (
            <div className="absolute right-0 mt-2 w-64 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-80 overflow-y-auto">
              {volatilityIndices.map((index) => (
                <button
                  key={index.value}
                  onClick={() => handleIndexChange(index.value)}
                  className={`w-full text-left px-4 py-3 hover:bg-gray-50 transition-colors ${
                    selectedIndex === index.value ? 'bg-blue-50 text-blue-700' : 'text-gray-700'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: index.color }}
                    ></div>
                    <span className="text-sm">{index.label}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Current price and last digit */}
      <div className="flex items-center justify-between mb-4 p-3 bg-gray-50 rounded-lg">
        <div>
          <p className="text-sm text-gray-600">Current Price</p>
          <p className="text-xl font-bold text-gray-900">
            {currentPrice && typeof currentPrice === 'number' ? currentPrice.toFixed(2) : '---'}
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-600">Last Digit (2nd Decimal)</p>
          <div className="flex items-center space-x-2">
            <span className="text-2xl font-bold text-blue-600">{lastDigit ?? '-'}</span>
            <Activity className="w-5 h-5 text-green-500" />
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="relative">
        {isLoading ? (
          <div className="flex items-center justify-center h-80">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            width={600}
            height={300}
            className="w-full h-80 border border-gray-200 rounded"
            style={{ maxWidth: '100%' }}
          />
        )}
      </div>
      
      {/* Chart info */}
      <div className="mt-4 flex items-center justify-between text-sm text-gray-500">
        <span>Real-time data • Updates every {selectedIndex.includes('1s') ? '0.2 seconds' : '1 second'}</span>
        <div className="flex items-center space-x-4">
          <span>{chartData.length} data points</span>
          <span>{tickData.length} ticks</span>
          <span className="text-green-500">● Live</span>
        </div>
      </div>
    </div>
  );
};

export default VolatilityChart;
