import React from 'react';
import { TrendingUp, BarChart2, Clock, AlertTriangle, ChevronRight } from 'lucide-react';

const StrategyCard = ({ strategy, isActive, onSelect, showDetails = false }) => {
  // Get risk level styling
  const getRiskLevelStyles = (riskLevel) => {
    const styles = {
      low: {
        bg: 'bg-green-100',
        text: 'text-green-800',
        border: 'border-green-200',
        icon: 'text-green-600',
        pill: 'bg-green-50 text-green-600 border-green-100'
      },
      medium: {
        bg: 'bg-blue-100',
        text: 'text-blue-800',
        border: 'border-blue-200',
        icon: 'text-blue-600',
        pill: 'bg-blue-50 text-blue-600 border-blue-100'
      },
      'medium-high': {
        bg: 'bg-amber-100',
        text: 'text-amber-800',
        border: 'border-amber-200',
        icon: 'text-amber-600',
        pill: 'bg-amber-50 text-amber-600 border-amber-100'
      },
      high: {
        bg: 'bg-red-100',
        text: 'text-red-800',
        border: 'border-red-200',
        icon: 'text-red-600',
        pill: 'bg-red-50 text-red-600 border-red-100'
      }
    };
    
    return styles[riskLevel] || styles.medium;
  };
  
  const styles = getRiskLevelStyles(strategy.risk_level);

  // Get the appropriate icon based on strategy name
  const getStrategyIcon = (strategyName) => {
    if (strategyName.includes('Momentum') || strategyName.includes('Trend')) {
      return <TrendingUp className={`w-5 h-5 ${styles.icon}`} />;
    } else if (strategyName.includes('RSI') || strategyName.includes('MACD')) {
      return <BarChart2 className={`w-5 h-5 ${styles.icon}`} />;
    } else if (strategyName.includes('Volatility')) {
      return <AlertTriangle className={`w-5 h-5 ${styles.icon}`} />;
    } else {
      return <Clock className={`w-5 h-5 ${styles.icon}`} />;
    }
  };

  return (
    <div 
      className={`p-4 rounded-lg border transition-all ${
        isActive 
          ? `${styles.border} ${styles.bg} shadow-sm` 
          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
      } ${showDetails ? 'cursor-pointer' : ''}`}
      onClick={showDetails ? () => onSelect(strategy) : null}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getStrategyIcon(strategy.name)}
          <h3 className={`font-medium ${isActive ? styles.text : 'text-gray-800'}`}>
            {strategy.name}
          </h3>
        </div>
        
        {isActive && (
          <span className="flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-600">
            Active
          </span>
        )}
      </div>
      
      <p className="text-sm text-gray-600 mb-3 line-clamp-2">
        {strategy.description}
      </p>
      
      <div className="flex items-center justify-between text-xs">
        <div className="flex items-center">
          <Clock className="w-3 h-3 text-gray-500 mr-1" />
          <span className="text-gray-600">{strategy.timeframe}</span>
        </div>
        
        <span className={`px-2 py-1 rounded-full ${styles.pill}`}>
          {strategy.risk_level.replace('-', ' ')}
        </span>
      </div>
      
      {showDetails && (
        <div className="mt-3 flex justify-end">
          <button className="flex items-center text-xs text-blue-600 hover:text-blue-800">
            View details
            <ChevronRight className="w-3 h-3 ml-1" />
          </button>
        </div>
      )}
    </div>
  );
};

export default StrategyCard;
