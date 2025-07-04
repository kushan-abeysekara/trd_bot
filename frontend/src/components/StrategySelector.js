import React, { useState, useEffect, useCallback } from 'react';
import { tradingAPI } from '../services/api';
import { 
  ChevronDown, ChevronUp, Check, AlertTriangle, Target, Info, 
  LineChart, Gauge, Zap, TrendingUp, BarChart2,
  Activity, RefreshCw, Clock
} from 'lucide-react';

const StrategySelector = ({ onStrategyChange, initialStrategies = null, initialSelectedStrategy = null, showDetails = false, showPerformance = false }) => {
  const [strategies, setStrategies] = useState(initialStrategies || []);
  const [activeStrategy, setActiveStrategy] = useState(initialSelectedStrategy || null);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showDetailsFor, setShowDetailsFor] = useState(null);
  const [strategyPerformance, setStrategyPerformance] = useState({});
  const [isLoadingPerformance, setIsLoadingPerformance] = useState(false);

  const fetchStrategies = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await tradingAPI.getStrategies();
      if (response.data && response.data.success) {
        setStrategies(response.data.strategies);
      }
    } catch (err) {
      setError('Failed to fetch strategies');
      console.error('Error fetching strategies:', err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchBotStatus = useCallback(async () => {
    try {
      const response = await tradingAPI.getBotStatus();
      if (response.data && response.data.success && response.data.status) {
        const currentStrategyName = response.data.status.current_strategy;
        const currentStrategy = strategies.find(s => s.name === currentStrategyName) || null;
        setActiveStrategy(currentStrategy);
      }
    } catch (err) {
      console.error('Error fetching bot status:', err);
    }
  }, [strategies]);

  // Fetch available strategies on component mount if not provided
  useEffect(() => {
    if (!initialStrategies) {
      fetchStrategies();
      // Fetch bot status to get current strategy
      fetchBotStatus();
    }
  }, [initialStrategies, fetchStrategies, fetchBotStatus]);

  // If showing performance metrics, fetch them
  useEffect(() => {
    if (showPerformance && strategies.length > 0) {
      fetchStrategyPerformance();
    }
  }, [showPerformance, strategies]);

  const fetchStrategyPerformance = async () => {
    try {
      setIsLoadingPerformance(true);
      const response = await tradingAPI.getStrategyPerformanceStats();
      if (response.data && response.data.success) {
        setStrategyPerformance(response.data.performance || {});
      }
    } catch (err) {
      console.error('Error fetching strategy performance:', err);
    } finally {
      setIsLoadingPerformance(false);
    }
  };

  // Group strategies by risk level
  const riskLevels = ['low', 'medium', 'medium-high', 'high'];
  const getRiskColor = (risk) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'medium': return 'text-blue-600 bg-blue-100';
      case 'medium-high': return 'text-amber-600 bg-amber-100';
      case 'high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const handleStrategySelect = async (strategy) => {
    try {
      setIsLoading(true);
      const response = await tradingAPI.setStrategy(strategy.id);
      if (response.data && response.data.success) {
        setActiveStrategy(strategy);
        if (onStrategyChange) {
          onStrategyChange(strategy);
        }
      } else {
        setError(response.data?.message || 'Failed to set strategy');
      }
    } catch (err) {
      setError('Failed to set strategy');
      console.error('Error setting strategy:', err);
    } finally {
      setIsLoading(false);
      setIsOpen(false);
    }
  };

  const getRiskBadge = (riskLevel) => {
    const riskColors = {
      'low': 'bg-green-100 text-green-800',
      'medium': 'bg-blue-100 text-blue-800',
      'medium-high': 'bg-amber-100 text-amber-800',
      'high': 'bg-orange-100 text-orange-800',
      'very_high': 'bg-red-100 text-red-800'
    };

    return (
      <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${riskColors[riskLevel] || 'bg-gray-100 text-gray-800'}`}>
        {riskLevel.replace('-', ' ')}
      </span>
    );
  };
  
  // Get performance metrics for a strategy
  /* Get performance metrics for a strategy - for future use
  const getPerformanceMetrics = (strategyId) => {
    const performance = strategyPerformance[strategyId] || {
      winRate: 0,
      totalTrades: 0,
      profitFactor: 0,
      averageProfit: 0
    };
    
    return performance;
  };
  */
  
  // Get icon for strategy based on name or properties
  const getStrategyIcon = (strategy) => {
    const name = strategy.name.toLowerCase();
    
    if (name.includes('momentum')) return <TrendingUp className="w-5 h-5" />;
    if (name.includes('rsi')) return <Activity className="w-5 h-5" />;
    if (name.includes('volatility')) return <Gauge className="w-5 h-5" />;
    if (name.includes('macd')) return <BarChart2 className="w-5 h-5" />;
    if (name.includes('bollinger') || name.includes('band')) return <LineChart className="w-5 h-5" />;
    if (name.includes('tick')) return <Clock className="w-5 h-5" />;
    if (name.includes('confluence')) return <Target className="w-5 h-5" />;
    
    return <Zap className="w-5 h-5" />;
  };
  
  // Group strategies by risk level for better organization
  const groupedStrategies = riskLevels.reduce((acc, riskLevel) => {
    acc[riskLevel] = strategies.filter(strategy => strategy.risk_level === riskLevel);
    return acc;
  }, {});

  return (
    <div className="relative">
      {/* Strategy selector dropdown button */}
      <div 
        className="bg-white border border-gray-200 rounded-lg shadow-sm hover:border-blue-300 transition-all cursor-pointer"
        onClick={() => !isLoading && setIsOpen(!isOpen)}
      >
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-full ${activeStrategy ? getRiskColor(activeStrategy.risk_level) : 'bg-gray-100'}`}>
              {activeStrategy ? getStrategyIcon(activeStrategy) : <Target className="w-5 h-5 text-gray-500" />}
            </div>
            <div>
              <p className="font-medium text-sm text-gray-900">Trading Strategy</p>
              <p className="text-base font-semibold text-gray-700">
                {activeStrategy ? activeStrategy.name : 'Select Strategy'}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {isLoading && (
              <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
            )}
            {isOpen ? <ChevronUp className="h-5 w-5 text-gray-500" /> : <ChevronDown className="h-5 w-5 text-gray-500" />}
          </div>
        </div>
        
        {showPerformance && activeStrategy && strategyPerformance[activeStrategy.id] && (
          <div className="border-t border-gray-100 px-4 py-2 grid grid-cols-4 gap-2 text-xs text-gray-600">
            <div className="flex flex-col items-center">
              <span className="font-medium">Win Rate</span>
              <span className={`font-semibold ${strategyPerformance[activeStrategy.id].winRate >= 50 ? 'text-green-600' : 'text-red-600'}`}>
                {strategyPerformance[activeStrategy.id].winRate}%
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-medium">Trades</span>
              <span className="font-semibold text-gray-700">
                {strategyPerformance[activeStrategy.id].totalTrades}
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-medium">Profit Factor</span>
              <span className={`font-semibold ${strategyPerformance[activeStrategy.id].profitFactor >= 1 ? 'text-green-600' : 'text-red-600'}`}>
                {strategyPerformance[activeStrategy.id].profitFactor.toFixed(2)}
              </span>
            </div>
            <div className="flex flex-col items-center">
              <span className="font-medium">Avg Profit</span>
              <span className={`font-semibold ${strategyPerformance[activeStrategy.id].averageProfit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {strategyPerformance[activeStrategy.id].averageProfit.toFixed(2)}%
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="absolute left-0 right-0 mt-2 bg-white border border-gray-200 rounded-lg shadow-lg z-50 max-h-[500px] overflow-y-auto">
          {error && (
            <div className="p-3 flex items-center bg-red-50 border-b border-red-100">
              <AlertTriangle className="w-4 h-4 text-red-500 mr-2" />
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          <div className="p-2">
            {riskLevels.map(riskLevel => {
              const strategiesInRisk = groupedStrategies[riskLevel] || [];
              if (strategiesInRisk.length === 0) return null;
              
              return (
                <div key={riskLevel} className="mb-2">
                  <div className="px-2 py-1">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                      {riskLevel.replace('-', ' ')} Risk
                    </h3>
                  </div>
                  
                  {strategiesInRisk.map(strategy => (
                    <div key={strategy.id} className="relative">
                      <div 
                        className={`px-3 py-2 hover:bg-gray-50 rounded-md flex items-center justify-between cursor-pointer ${
                          activeStrategy?.id === strategy.id ? 'bg-blue-50' : ''
                        }`}
                        onClick={() => handleStrategySelect(strategy)}
                      >
                        <div className="flex items-center space-x-3">
                          <div className={`p-1.5 rounded-full ${getRiskColor(strategy.risk_level)}`}>
                            {getStrategyIcon(strategy)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className={`text-sm font-medium ${activeStrategy?.id === strategy.id ? 'text-blue-600' : 'text-gray-700'}`}>
                              {strategy.name}
                            </p>
                            <p className="text-xs text-gray-500 truncate max-w-xs">
                              {strategy.description}
                            </p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {showPerformance && !isLoadingPerformance && strategyPerformance[strategy.id] && (
                            <div className="text-xs text-gray-500 flex items-center space-x-1">
                              <span className={strategyPerformance[strategy.id].winRate >= 50 ? 'text-green-600' : 'text-red-600'}>
                                {strategyPerformance[strategy.id].winRate}% 
                              </span>
                              <span>win rate</span>
                            </div>
                          )}
                          
                          {activeStrategy?.id === strategy.id && (
                            <Check className="w-5 h-5 text-blue-600" />
                          )}
                          
                          {showDetails && (
                            <button
                              type="button"
                              className="p-1 rounded-full hover:bg-gray-200 text-gray-500"
                              onClick={(e) => {
                                e.stopPropagation();
                                setShowDetailsFor(showDetailsFor === strategy.id ? null : strategy.id);
                              }}
                            >
                              <Info className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </div>
                      
                      {/* Strategy Details Panel */}
                      {showDetailsFor === strategy.id && (
                        <div className="bg-gray-50 px-3 py-2 mx-2 mb-2 rounded-md text-sm border-l-2 border-blue-400">
                          <div className="mb-2">
                            <span className="font-medium text-gray-700">Description:</span> 
                            <p className="text-gray-600">{strategy.description}</p>
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <span className="font-medium text-gray-700">Risk Level:</span> 
                              <span className="ml-1">{getRiskBadge(strategy.risk_level)}</span>
                            </div>
                            <div>
                              <span className="font-medium text-gray-700">Timeframe:</span> 
                              <span className="ml-1 text-gray-600">{strategy.timeframe}</span>
                            </div>
                            
                            {showPerformance && strategyPerformance[strategy.id] && (
                              <>
                                <div>
                                  <span className="font-medium text-gray-700">Total Trades:</span> 
                                  <span className="ml-1 text-gray-600">{strategyPerformance[strategy.id].totalTrades}</span>
                                </div>
                                <div>
                                  <span className="font-medium text-gray-700">Win Rate:</span> 
                                  <span className={`ml-1 ${strategyPerformance[strategy.id].winRate >= 50 ? 'text-green-600' : 'text-red-600'}`}>
                                    {strategyPerformance[strategy.id].winRate}%
                                  </span>
                                </div>
                                <div>
                                  <span className="font-medium text-gray-700">Profit Factor:</span> 
                                  <span className={`ml-1 ${strategyPerformance[strategy.id].profitFactor >= 1 ? 'text-green-600' : 'text-red-600'}`}>
                                    {strategyPerformance[strategy.id].profitFactor.toFixed(2)}
                                  </span>
                                </div>
                                <div>
                                  <span className="font-medium text-gray-700">Avg. Profit:</span> 
                                  <span className={`ml-1 ${strategyPerformance[strategy.id].averageProfit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                                    {strategyPerformance[strategy.id].averageProfit.toFixed(2)}%
                                  </span>
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
          
          {/* Footer actions */}
          <div className="border-t border-gray-100 px-3 py-2 flex items-center justify-between text-xs">
            <button 
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                fetchStrategies();
                if (showPerformance) fetchStrategyPerformance();
              }}
              className="text-blue-600 hover:text-blue-800 flex items-center"
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              Refresh
            </button>
            <span className="text-gray-500">
              {strategies.length} strategies available
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategySelector;
