import React, { useState } from 'react';
import { Activity, ArrowUp, ArrowDown, Clock, Check, X, DollarSign, Info } from 'lucide-react';

const ActiveTradesTable = ({ trades = [], onForceClose }) => {
  const [showDetails, setShowDetails] = useState(null);

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${seconds}s`;
    return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  const calculateTimeRemaining = (trade) => {
    const entryTime = new Date(trade.entry_time).getTime();
    const duration = trade.duration * 1000; // Convert seconds to ms
    const endTime = entryTime + duration;
    const remaining = Math.max(0, Math.floor((endTime - Date.now()) / 1000));
    return formatDuration(remaining);
  };

  const getProgressPercentage = (trade) => {
    const entryTime = new Date(trade.entry_time).getTime();
    const duration = trade.duration * 1000; // Convert seconds to ms
    const elapsed = Date.now() - entryTime;
    return Math.min(100, Math.max(0, (elapsed / duration) * 100));
  };

  if (trades.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 bg-gray-50 rounded-lg border border-gray-200">
        <Activity className="w-10 h-10 text-gray-300 mb-2" />
        <p className="text-gray-500">No active trades</p>
        <p className="text-xs text-gray-400 mt-1">Trades will appear here when the bot makes a new entry</p>
      </div>
    );
  }

  return (
    <div className="overflow-hidden bg-white rounded-lg border border-gray-200">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Direction
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Entry Price
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Stake
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Remaining
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Time
            </th>
            <th scope="col" className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {trades.map((trade) => (
            <React.Fragment key={trade.id}>
              <tr className={`${showDetails === trade.id ? 'bg-blue-50' : 'hover:bg-gray-50'}`}>
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="flex items-center">
                    {trade.direction === 'RISE' ? (
                      <ArrowUp className="flex-shrink-0 w-4 h-4 text-green-500 mr-1.5" />
                    ) : (
                      <ArrowDown className="flex-shrink-0 w-4 h-4 text-red-500 mr-1.5" />
                    )}
                    <span className={`text-sm font-medium ${trade.direction === 'RISE' ? 'text-green-600' : 'text-red-600'}`}>
                      {trade.direction}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="text-sm text-gray-900">${trade.entry_price.toFixed(5)}</div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="flex items-center text-sm text-gray-900">
                    <DollarSign className="w-3.5 h-3.5 text-gray-500 mr-1" />
                    {trade.stake.toFixed(2)}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="w-full">
                    <div className="flex items-center text-xs mb-1">
                      <Clock className="w-3 h-3 text-gray-400 mr-1" />
                      <span className="text-gray-600">{calculateTimeRemaining(trade)}</span>
                    </div>
                    <div className="w-20 bg-gray-200 rounded-full h-1.5">
                      <div 
                        className="bg-blue-500 h-1.5 rounded-full" 
                        style={{width: `${getProgressPercentage(trade)}%`}}
                      ></div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <div className="text-sm text-gray-500 truncate max-w-xs" title={trade.strategy}>
                    {trade.strategy}
                  </div>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex justify-end space-x-2">
                    <button
                      onClick={() => setShowDetails(showDetails === trade.id ? null : trade.id)}
                      className="text-blue-600 hover:text-blue-800"
                    >
                      <Info className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => onForceClose(trade.id)}
                      className="text-red-600 hover:text-red-800"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>

              {/* Details row */}
              {showDetails === trade.id && (
                <tr className="bg-blue-50">
                  <td colSpan="6" className="px-4 py-3">
                    <div className="grid grid-cols-3 gap-4 text-xs">
                      <div>
                        <span className="text-gray-500">Trade ID:</span>
                        <div className="font-medium text-gray-900 mt-1">{trade.id}</div>
                      </div>
                      <div>
                        <span className="text-gray-500">Entry Time:</span>
                        <div className="font-medium text-gray-900 mt-1">
                          {new Date(trade.entry_time).toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <span className="text-gray-500">Duration:</span>
                        <div className="font-medium text-gray-900 mt-1">{trade.duration} seconds</div>
                      </div>
                      <div className="col-span-3">
                        <span className="text-gray-500">Strategy Reason:</span>
                        <div className="font-medium text-gray-900 mt-1">{trade.reason || 'No reason provided'}</div>
                      </div>
                      <div className="col-span-3 mt-2">
                        <button
                          onClick={() => onForceClose(trade.id)}
                          className="flex items-center px-3 py-1 bg-red-100 text-red-700 text-xs rounded-md hover:bg-red-200"
                        >
                          <X className="w-3 h-3 mr-1" />
                          Force Close Trade
                        </button>
                      </div>
                    </div>
                  </td>
                </tr>
              )}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ActiveTradesTable;
