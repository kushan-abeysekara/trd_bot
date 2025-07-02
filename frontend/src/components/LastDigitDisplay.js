import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp } from 'lucide-react';

const LastDigitDisplay = ({ lastDigit, indexName }) => {
  const [digitHistory, setDigitHistory] = useState([]);
  const [isActive, setIsActive] = useState(false);

  useEffect(() => {
    if (lastDigit !== null && lastDigit !== undefined) {
      setDigitHistory(prev => {
        const newHistory = [lastDigit, ...prev.slice(0, 9)]; // Keep last 10 digits
        return newHistory;
      });
      
      // Animate the update
      setIsActive(true);
      const timer = setTimeout(() => setIsActive(false), 500);
      return () => clearTimeout(timer);
    }
  }, [lastDigit]);

  const getDigitColor = (digit) => {
    const colors = [
      'text-red-600',    // 0
      'text-blue-600',   // 1
      'text-green-600',  // 2
      'text-yellow-600', // 3
      'text-purple-600', // 4
      'text-pink-600',   // 5
      'text-indigo-600', // 6
      'text-orange-600', // 7
      'text-teal-600',   // 8
      'text-cyan-600'    // 9
    ];
    return colors[digit] || 'text-gray-600';
  };

  if (lastDigit === null || lastDigit === undefined) {
    return null;
  }

  return (
    <div className="flex items-center space-x-3 px-4 py-2 bg-white border border-gray-200 rounded-lg shadow-sm">
      <div className="flex items-center space-x-2">
        <Activity className={`w-4 h-4 ${isActive ? 'text-green-500 animate-pulse' : 'text-gray-400'}`} />
        <span className="text-xs text-gray-600 font-medium">Live</span>
      </div>
      
      <div className="h-6 w-px bg-gray-300"></div>
      
      <div className="flex items-center space-x-2">
        <span className="text-xs text-gray-500">2nd Decimal:</span>
        <span className={`text-lg font-bold ${getDigitColor(lastDigit)} ${isActive ? 'animate-bounce' : ''}`}>
          {lastDigit}
        </span>
      </div>
      
      <div className="h-6 w-px bg-gray-300"></div>
      
      {/* Digit history */}
      <div className="flex items-center space-x-1">
        <span className="text-xs text-gray-400">History:</span>
        <div className="flex space-x-1">
          {digitHistory.slice(1, 6).map((digit, index) => (
            <span 
              key={index}
              className={`text-xs font-mono ${getDigitColor(digit)} opacity-${90 - (index * 15)}`}
            >
              {digit}
            </span>
          ))}
        </div>
      </div>
      
      {indexName && (
        <>
          <div className="h-6 w-px bg-gray-300"></div>
          <div className="flex items-center space-x-1">
            <TrendingUp className="w-3 h-3 text-blue-500" />
            <span className="text-xs text-gray-600 font-medium truncate max-w-32" title={indexName}>
              {indexName.replace(' Index', '')}
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default LastDigitDisplay;
