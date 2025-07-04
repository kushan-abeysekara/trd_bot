import React, { useState, useEffect } from 'react';
import { Gauge, TrendingUp, TrendingDown, BarChart2, Activity } from 'lucide-react';

const TradingIndicators = ({ indicators, chartData, lastSignal }) => {
  const [lastUpdate, setLastUpdate] = useState(Date.now());
  
  // Update the last update timestamp
  useEffect(() => {
    setLastUpdate(Date.now());
  }, [indicators]);

  const formatTimeAgo = (timestamp) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 5) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ago`;
  };

  // Calculate indicator status
  const getRSIStatus = (value) => {
    if (value > 70) return { color: 'text-red-600', status: 'Overbought' };
    if (value < 30) return { color: 'text-green-600', status: 'Oversold' };
    return { color: 'text-gray-600', status: 'Neutral' };
  };

  const getMACDStatus = (macd, signal) => {
    if (macd > signal) return { color: 'text-green-600', status: 'Bullish' };
    return { color: 'text-red-600', status: 'Bearish' };
  };

  const getMomentumStatus = (value) => {
    if (value > 0.1) return { color: 'text-green-600', status: 'Strong Up' };
    if (value < -0.1) return { color: 'text-red-600', status: 'Strong Down' };
    if (value > 0) return { color: 'text-green-500', status: 'Slight Up' };
    if (value < 0) return { color: 'text-red-500', status: 'Slight Down' };
    return { color: 'text-gray-600', status: 'Flat' };
  };

  const getVolatilityStatus = (value) => {
    const percent = value * 100;
    if (percent > 1.5) return { color: 'text-red-600', status: 'High' };
    if (percent > 0.8) return { color: 'text-amber-600', status: 'Medium' };
    return { color: 'text-green-600', status: 'Low' };
  };

  const rsiStatus = getRSIStatus(indicators.rsi);
  const macdStatus = getMACDStatus(indicators.macd, indicators.macdSignal);
  const momentumStatus = getMomentumStatus(indicators.momentum);
  const volatilityStatus = getVolatilityStatus(indicators.volatility);

  // Calculate percentages for gauge visualizations
  const rsiPercentage = Math.min(Math.max(indicators.rsi, 0), 100);
  const volatilityPercentage = Math.min(Math.max(indicators.volatility * 100 * 3, 0), 100); // Scale for better visibility

  return (
    <div className="bg-white rounded-lg shadow p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center">
          <BarChart2 className="w-5 h-5 mr-2 text-blue-600" />
          Technical Indicators
        </h3>
        <div className="text-xs text-gray-500 flex items-center">
          <div className="flex items-center">
            <div className="w-2 h-2 rounded-full bg-green-500 mr-1"></div>
            <span>Updated {formatTimeAgo(lastUpdate)}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* RSI Indicator */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex justify-between mb-2">
            <div className="text-sm font-medium text-gray-700">RSI</div>
            <div className={`text-sm font-medium ${rsiStatus.color}`}>{rsiStatus.status}</div>
          </div>
          
          <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden mb-1">
            <div 
              className={`absolute left-0 top-0 bottom-0 ${
                rsiPercentage > 70 ? 'bg-red-500' : 
                rsiPercentage < 30 ? 'bg-green-500' : 'bg-blue-500'
              }`}
              style={{ width: `${rsiPercentage}%` }}
            ></div>
          </div>
          
          <div className="flex justify-between text-xs text-gray-500">
            <div>0</div>
            <div>30</div>
            <div>70</div>
            <div>100</div>
          </div>
          
          <div className="mt-2">
            <span className="text-lg font-semibold">{indicators.rsi.toFixed(1)}</span>
          </div>
        </div>

        {/* MACD Indicator */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex justify-between mb-2">
            <div className="text-sm font-medium text-gray-700">MACD</div>
            <div className={`text-sm font-medium ${macdStatus.color}`}>{macdStatus.status}</div>
          </div>
          
          <div className="flex items-center space-x-2 mb-2">
            <div className="flex items-center">
              <div className="w-3 h-2 bg-blue-500 mr-1"></div>
              <span className="text-xs text-gray-600">MACD</span>
            </div>
            <div className="flex items-center">
              <div className="w-3 h-2 bg-red-400 mr-1"></div>
              <span className="text-xs text-gray-600">Signal</span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-2">
            <div className="p-2 bg-white rounded">
              <div className="text-xs text-gray-500">MACD</div>
              <div className={`text-base font-medium ${indicators.macd > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {indicators.macd.toFixed(3)}
              </div>
            </div>
            <div className="p-2 bg-white rounded">
              <div className="text-xs text-gray-500">Signal</div>
              <div className="text-base font-medium text-gray-700">
                {indicators.macdSignal.toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Momentum Indicator */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex justify-between mb-2">
            <div className="text-sm font-medium text-gray-700">Momentum</div>
            <div className={`text-sm font-medium ${momentumStatus.color}`}>{momentumStatus.status}</div>
          </div>
          
          <div className="flex items-center">
            {indicators.momentum < 0 ? (
              <TrendingDown className={`w-5 h-5 mr-2 ${momentumStatus.color}`} />
            ) : (
              <TrendingUp className={`w-5 h-5 mr-2 ${momentumStatus.color}`} />
            )}
            <span className={`text-lg font-semibold ${momentumStatus.color}`}>
              {(indicators.momentum * 100).toFixed(2)}%
            </span>
          </div>
        </div>

        {/* Volatility Indicator */}
        <div className="p-3 bg-gray-50 rounded-lg">
          <div className="flex justify-between mb-2">
            <div className="text-sm font-medium text-gray-700">Volatility</div>
            <div className={`text-sm font-medium ${volatilityStatus.color}`}>{volatilityStatus.status}</div>
          </div>
          
          <div className="relative h-3 bg-gray-200 rounded-full overflow-hidden mb-1">
            <div 
              className={`absolute left-0 top-0 bottom-0 ${
                volatilityPercentage > 70 ? 'bg-red-500' : 
                volatilityPercentage > 40 ? 'bg-amber-500' : 'bg-green-500'
              }`}
              style={{ width: `${volatilityPercentage}%` }}
            ></div>
          </div>
          
          <div className="mt-1">
            <span className="text-lg font-semibold">{(indicators.volatility * 100).toFixed(2)}%</span>
          </div>
        </div>
      </div>

      {/* Latest Signal */}
      {lastSignal && (
        <div className="mt-4 p-3 rounded-lg bg-blue-50 border border-blue-100">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium text-blue-800">Latest Signal</div>
            <div className="text-xs text-blue-600">{formatTimeAgo(lastSignal.timestamp)}</div>
          </div>
          <div className="flex items-center mt-1">
            <Activity className="w-4 h-4 mr-1 text-blue-600" />
            <div className="text-sm">
              <span className={`font-medium ${lastSignal.direction === 'RISE' ? 'text-green-600' : 'text-red-600'}`}>
                {lastSignal.direction}
              </span>
              <span className="mx-1 text-gray-600">·</span>
              <span className="text-gray-700">{lastSignal.strategy}</span>
              <span className="mx-1 text-gray-600">·</span>
              <span className="text-gray-700">{lastSignal.confidence}% confidence</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradingIndicators;
