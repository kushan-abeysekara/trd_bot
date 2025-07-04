import React from 'react';
import { 
  ArrowUp, ArrowDown, Clock, AlertCircle, 
  TrendingUp, TrendingDown, CheckCircle2, XCircle
} from 'lucide-react';

const TradeSignalPanel = ({ 
  latestSignal, 
  recentSignals = [], 
  activeTrades = [], 
  tradeHistory = [], 
  loading = false 
}) => {
  
  // Format timestamp to readable time
  const formatTime = (timestamp) => {
    if (!timestamp) return '';
    
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };
  
  // Calculate time elapsed since a timestamp
  const timeElapsed = (timestamp) => {
    if (!timestamp) return '';
    
    const now = new Date();
    const signalTime = new Date(timestamp);
    const seconds = Math.floor((now - signalTime) / 1000);
    
    if (seconds < 60) {
      return `${seconds}s ago`;
    } else if (seconds < 3600) {
      return `${Math.floor(seconds / 60)}m ago`;
    } else {
      return `${Math.floor(seconds / 3600)}h ago`;
    }
  };
  
  // Get class for trade direction
  const getDirectionClass = (direction) => {
    return direction === 'RISE' 
      ? 'text-green-600 bg-green-50' 
      : 'text-red-600 bg-red-50';
  };
  
  // Get icon for trade direction
  const getDirectionIcon = (direction) => {
    return direction === 'RISE' 
      ? <ArrowUp className="w-4 h-4" /> 
      : <ArrowDown className="w-4 h-4" />;
  };
  
  // Get status badge for trade
  const getStatusBadge = (status) => {
    switch (status) {
      case 'WON':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <CheckCircle2 className="w-3 h-3 mr-0.5" />
            Won
          </span>
        );
      case 'LOST':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
            <XCircle className="w-3 h-3 mr-0.5" />
            Lost
          </span>
        );
      case 'ACTIVE':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            <Clock className="w-3 h-3 mr-0.5" />
            Active
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
            {status}
          </span>
        );
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md border border-gray-100 overflow-hidden">
      {/* Panel header */}
      <div className="border-b border-gray-100 px-4 py-3">
        <h2 className="font-semibold text-gray-800">Trading Signals</h2>
        <p className="text-xs text-gray-500 mt-1">Real-time trade signals from active strategies</p>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="p-6 flex items-center justify-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Loading signals...</span>
        </div>
      )}

      {/* Latest signal */}
      <div className="px-4 py-3 border-b border-gray-100">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-sm font-medium text-gray-700">Latest Signal</h3>
          {latestSignal && (
            <span className="text-xs text-gray-500 flex items-center">
              <Clock className="w-3 h-3 mr-1" />
              {formatTime(latestSignal.timestamp)}
            </span>
          )}
        </div>

        {!latestSignal ? (
          <div className="flex items-center justify-center py-4 text-gray-500">
            <AlertCircle className="w-5 h-5 mr-2" />
            <span>No signals detected yet</span>
          </div>
        ) : (
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <div className={`inline-flex items-center px-2 py-1 rounded-md text-sm font-medium ${getDirectionClass(latestSignal.direction)}`}>
                {getDirectionIcon(latestSignal.direction)}
                <span className="ml-1">{latestSignal.direction}</span>
              </div>
              <div className="text-sm font-medium text-gray-700">
                Confidence: {latestSignal.confidence}%
              </div>
            </div>
            
            <div className="text-sm text-gray-600 mb-2">
              <p className="truncate">{latestSignal.reason}</p>
            </div>
            
            <div className="flex justify-between items-center text-xs text-gray-500">
              <div>Price: {latestSignal.entry_price}</div>
              <div>Duration: {latestSignal.duration}s</div>
            </div>
          </div>
        )}
      </div>

      {/* Active trades */}
      <div className="px-4 py-3 border-b border-gray-100">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Active Trades ({activeTrades.length})</h3>
        
        {activeTrades.length === 0 ? (
          <p className="text-center text-gray-500 py-2 text-sm">No active trades</p>
        ) : (
          <div className="space-y-2">
            {activeTrades.slice(0, 3).map(trade => (
              <div key={trade.id} className="flex justify-between items-center bg-gray-50 rounded p-2 text-sm">
                <div className="flex items-center">
                  <div className={`p-1 rounded ${trade.direction === 'RISE' ? 'bg-green-100' : 'bg-red-100'} mr-2`}>
                    {trade.direction === 'RISE' ? 
                      <TrendingUp className={`w-4 h-4 ${trade.direction === 'RISE' ? 'text-green-600' : 'text-red-600'}`} /> : 
                      <TrendingDown className={`w-4 h-4 ${trade.direction === 'RISE' ? 'text-green-600' : 'text-red-600'}`} />
                    }
                  </div>
                  <div className="flex flex-col">
                    <span className="font-medium">{trade.direction} @ ${trade.entry_price}</span>
                    <span className="text-xs text-gray-500">{formatTime(trade.entry_time)}</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium">${trade.stake}</div>
                  <div className="text-xs text-gray-500">{trade.duration}s duration</div>
                </div>
              </div>
            ))}
            
            {activeTrades.length > 3 && (
              <div className="text-center text-xs text-blue-600 py-1">
                +{activeTrades.length - 3} more active trades
              </div>
            )}
          </div>
        )}
      </div>

      {/* Recent signals */}
      <div className="px-4 py-3">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Recent Signals</h3>
        
        {recentSignals.length === 0 ? (
          <p className="text-center text-gray-500 py-2 text-sm">No recent signals</p>
        ) : (
          <div className="divide-y divide-gray-100">
            {recentSignals.slice(0, 5).map((signal, index) => (
              <div key={index} className="py-2 flex justify-between items-center">
                <div className="flex items-center">
                  <div className={`p-1 rounded-full ${getDirectionClass(signal.direction)} mr-2`}>
                    {getDirectionIcon(signal.direction)}
                  </div>
                  <div className="text-sm">
                    <p className="font-medium text-gray-700">{signal.direction}</p>
                    <p className="text-xs text-gray-500 truncate max-w-[180px]">{signal.reason}</p>
                  </div>
                </div>
                <div className="text-xs text-gray-500">
                  {timeElapsed(signal.timestamp)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent trade history */}
      <div className="border-t border-gray-100 px-4 py-3 bg-gray-50">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Recent Results</h3>
        
        {tradeHistory.length === 0 ? (
          <p className="text-center text-gray-500 py-2 text-sm">No trade history</p>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            {tradeHistory.slice(0, 4).map(trade => (
              <div key={trade.id} className="flex items-center p-2 bg-white rounded border border-gray-100">
                <div className="mr-2">
                  {trade.status === 'WON' ? (
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                  ) : trade.status === 'LOST' ? (
                    <XCircle className="w-4 h-4 text-red-500" />
                  ) : (
                    <Clock className="w-4 h-4 text-gray-400" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-700 truncate">
                    {trade.direction} ${trade.stake}
                  </p>
                  <p className={`text-xs ${
                    trade.profit_loss > 0 ? 'text-green-600' : 
                    trade.profit_loss < 0 ? 'text-red-600' : 'text-gray-500'
                  }`}>
                    {trade.profit_loss > 0 ? '+' : ''}{trade.profit_loss}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeSignalPanel;
