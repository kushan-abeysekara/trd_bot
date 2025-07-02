import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp } from 'lucide-react';

const LastDigitDisplay = ({ lastDigit, indexName }) => {
  const [digitHistory, setDigitHistory] = useState([]);
  const [isActive, setIsActive] = useState(false);
  const [digitPattern, setDigitPattern] = useState(null);
  const [frequencyAnalysis, setFrequencyAnalysis] = useState({});

  useEffect(() => {
    if (lastDigit !== null && lastDigit !== undefined) {
      setDigitHistory(prev => {
        const newHistory = [lastDigit, ...prev.slice(0, 49)]; // Keep last 50 digits
        
        // Analyze patterns
        analyzeDigitPatterns(newHistory);
        analyzeFrequency(newHistory);
        
        return newHistory;
      });
      
      // Animate the update
      setIsActive(true);
      const timer = setTimeout(() => setIsActive(false), 500);
      return () => clearTimeout(timer);
    }
  }, [lastDigit]);

  const analyzeDigitPatterns = (history) => {
    if (history.length < 10) return;
    
    // Look for repeating patterns
    const patterns = {};
    for (let len = 2; len <= 5; len++) {
      for (let i = 0; i <= history.length - len * 2; i++) {
        const pattern = history.slice(i, i + len).join('');
        const nextPattern = history.slice(i + len, i + len * 2).join('');
        if (pattern === nextPattern) {
          patterns[pattern] = (patterns[pattern] || 0) + 1;
        }
      }
    }
    
    // Find most frequent pattern
    const mostFrequent = Object.entries(patterns).reduce((max, [pattern, count]) => 
      count > max.count ? { pattern, count } : max, { pattern: null, count: 0 });
    
    setDigitPattern(mostFrequent);
  };

  const analyzeFrequency = (history) => {
    const frequency = {};
    history.forEach(digit => {
      frequency[digit] = (frequency[digit] || 0) + 1;
    });
    setFrequencyAnalysis(frequency);
  };

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
    return (
      <div className="flex items-center space-x-3 px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg shadow-sm min-w-[280px] h-12">
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-gray-400" />
          <span className="text-xs text-gray-400 font-medium">Waiting...</span>
        </div>
        <div className="h-6 w-px bg-gray-300"></div>
        <div className="flex items-center space-x-2">
          <span className="text-xs text-gray-400">Last Digit:</span>
          <span className="text-lg font-bold text-gray-400">-</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center space-x-3 px-4 py-2 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-all duration-200 min-w-[280px] h-12">
      <div className="flex items-center space-x-2 min-w-[60px]">
        <Activity className={`w-4 h-4 transition-colors ${isActive ? 'text-green-500 animate-pulse' : 'text-gray-400'}`} />
        <span className={`text-xs font-medium transition-colors ${isActive ? 'text-green-600' : 'text-gray-600'}`}>
          {isActive ? 'Live' : 'Idle'}
        </span>
      </div>
      
      <div className="h-6 w-px bg-gray-300"></div>
      
      <div className="flex items-center space-x-2 min-w-[80px]">
        <span className="text-xs text-gray-500">Last Digit:</span>
        <span className={`text-lg font-bold transition-all duration-300 ${getDigitColor(lastDigit)} ${
          isActive ? 'transform scale-110' : ''
        }`}>
          {lastDigit}
        </span>
      </div>
      
      <div className="h-6 w-px bg-gray-300"></div>
      
      {/* Enhanced digit history with fixed width */}
      <div className="flex items-center space-x-1 min-w-[120px]">
        <span className="text-xs text-gray-400">History:</span>
        <div className="flex space-x-1 overflow-hidden">
          {digitHistory.slice(1, 8).map((digit, index) => (
            <span 
              key={index}
              className={`text-xs font-mono transition-all duration-200 ${getDigitColor(digit)}`}
              style={{ opacity: Math.max(0.3, 1 - (index * 0.15)) }}
              title={`${frequencyAnalysis[digit] || 0} times`}
            >
              {digit}
            </span>
          ))}
        </div>
      </div>
      
      {/* Pattern detection with fixed width */}
      {digitPattern?.pattern && (
        <>
          <div className="h-6 w-px bg-gray-300"></div>
          <div className="flex items-center space-x-1 min-w-[80px]">
            <span className="text-xs text-orange-500">Pattern:</span>
            <span className="text-xs font-mono text-orange-600 bg-orange-50 px-2 py-1 rounded-full">
              {digitPattern.pattern}
            </span>
            <span className="text-xs text-gray-400">({digitPattern.count}x)</span>
          </div>
        </>
      )}
      
      {indexName && (
        <>
          <div className="h-6 w-px bg-gray-300"></div>
          <div className="flex items-center space-x-1 min-w-[100px]">
            <TrendingUp className="w-3 h-3 text-blue-500" />
            <span className="text-xs text-gray-600 font-medium truncate max-w-[80px]" title={indexName}>
              {indexName.replace(' Index', '').replace('Volatility', 'Vol')}
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default LastDigitDisplay;
