import React, { useState, useEffect } from 'react';
import { X, Key, Shield, AlertTriangle, Eye, EyeOff } from 'lucide-react';
import { tradingAPI } from '../services/api';
import toast from 'react-hot-toast';

const ApiTokenSetup = ({ isOpen, onClose, onSuccess, user, currentAccountType }) => {
  const [demoApiToken, setDemoApiToken] = useState('');
  const [realApiToken, setRealApiToken] = useState('');
  const [activeTab, setActiveTab] = useState('demo');
  const [isLoading, setIsLoading] = useState(false);
  const [showDemoToken, setShowDemoToken] = useState(false);
  const [showRealToken, setShowRealToken] = useState(false);

  // Check if tokens are already saved
  const hasDemoToken = user?.has_demo_token || (user?.deriv_account_type === 'demo' && user?.has_api_token);
  const hasRealToken = user?.has_real_token || (user?.deriv_account_type === 'real' && user?.has_api_token);

  // Load existing tokens when component opens
  useEffect(() => {
    if (isOpen && user) {
      // Don't show any token values for saved tokens, keep fields empty
      setDemoApiToken('');
      setRealApiToken('');
    }
  }, [isOpen, user, hasDemoToken, hasRealToken]);

  // Set active tab based on current account type when opening
  useEffect(() => {
    if (isOpen && currentAccountType) {
      setActiveTab(currentAccountType);
    }
  }, [isOpen, currentAccountType]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const currentToken = activeTab === 'demo' ? demoApiToken : realApiToken;
    const hasCurrentToken = activeTab === 'demo' ? hasDemoToken : hasRealToken;
    
    if (!currentToken.trim()) {
      if (hasCurrentToken) {
        toast.error(`Please enter a new ${activeTab.toUpperCase()} API token to update the existing one`);
      } else {
        toast.error(`Please enter your ${activeTab.toUpperCase()} API token`);
      }
      return;
    }

    setIsLoading(true);
    try {
      await tradingAPI.setupApiToken({
        api_token: currentToken.trim(),
        account_type: activeTab
      });
      
      const actionText = hasCurrentToken ? 'Updated' : 'Setup';
      toast.success(`${activeTab.toUpperCase()} API Token ${actionText} Successfully!`);
      
      // Clear the form
      if (activeTab === 'demo') setDemoApiToken('');
      if (activeTab === 'real') setRealApiToken('');
      
      // Refresh user data
      await onSuccess();
      
      // Auto-close if this was the current account type setup
      if (activeTab === currentAccountType) {
        setTimeout(() => {
          onClose();
        }, 1500);
      }
    } catch (error) {
      toast.error(error.response?.data?.error || `Failed to ${hasCurrentToken ? 'update' : 'setup'} ${activeTab.toUpperCase()} API token`);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleTokenVisibility = (tokenType) => {
    if (tokenType === 'demo') {
      setShowDemoToken(!showDemoToken);
    } else {
      setShowRealToken(!showRealToken);
    }
  };

  const clearTokenField = (tokenType) => {
    if (tokenType === 'demo') {
      setDemoApiToken('');
    } else {
      setRealApiToken('');
    }
  };

  if (!isOpen) return null;

  const currentToken = activeTab === 'demo' ? demoApiToken : realApiToken;
  const hasCurrentToken = activeTab === 'demo' ? hasDemoToken : hasRealToken;
  const showCurrentToken = activeTab === 'demo' ? showDemoToken : showRealToken;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-900">
            {hasDemoToken || hasRealToken ? 'Manage' : 'Setup'} Deriv API Tokens
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="mb-6">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Key className="w-8 h-8 text-green-600" />
          </div>
          <p className="text-gray-600 text-center mb-4">
            {hasDemoToken || hasRealToken 
              ? 'Update your existing API tokens or configure new ones'
              : 'Configure both Demo and Real API tokens for flexible trading'
            }
          </p>
          
          {/* Warning */}
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-4">
            <div className="flex items-start space-x-2">
              <AlertTriangle className="w-5 h-5 text-amber-600 mt-0.5" />
              <div className="text-sm text-amber-800">
                <p className="font-medium">Security Notice:</p>
                <p>Your API tokens are encrypted and stored securely. Only use tokens with limited permissions.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex mb-4 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setActiveTab('demo')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors relative ${
              activeTab === 'demo' 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Demo Account
            {hasDemoToken && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
            )}
          </button>
          <button
            onClick={() => setActiveTab('real')}
            className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors relative ${
              activeTab === 'real' 
                ? 'bg-red-600 text-white' 
                : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Real Account
            {hasRealToken && (
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
            )}
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* API Token Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              {activeTab === 'demo' ? 'Demo' : 'Real'} API Token
              {hasCurrentToken && (
                <span className="ml-2 text-xs text-green-600 font-medium">(Saved)</span>
              )}
            </label>
            
            {/* Show status for saved tokens */}
            {hasCurrentToken && (
              <div className="mb-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Shield className="w-4 h-4 text-green-600" />
                    <span className="text-sm text-green-800 font-medium">
                      {activeTab.toUpperCase()} API Token is configured
                    </span>
                  </div>
                  <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded font-medium">
                    Active
                  </span>
                </div>
                <p className="text-xs text-green-700 mt-1">
                  Enter a new token below to update the existing configuration
                </p>
              </div>
            )}
            
            <div className="relative">
              <Shield className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type={showCurrentToken ? 'text' : 'password'}
                value={currentToken}
                onChange={(e) => {
                  if (activeTab === 'demo') {
                    setDemoApiToken(e.target.value);
                  } else {
                    setRealApiToken(e.target.value);
                  }
                }}
                className="w-full pl-10 pr-12 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder={
                  hasCurrentToken 
                    ? `Enter new ${activeTab.toUpperCase()} token to update`
                    : `Enter your ${activeTab.toUpperCase()} Deriv API token`
                }
                required
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <button
                  type="button"
                  onClick={() => toggleTokenVisibility(activeTab)}
                  className="text-gray-400 hover:text-gray-600"
                  title={showCurrentToken ? 'Hide token' : 'Show token'}
                >
                  {showCurrentToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Get your {activeTab.toUpperCase()} API token from Deriv app → Settings → API Token
            </p>
          </div>

          {/* Account Type Info */}
          <div className={`p-3 rounded-lg ${
            activeTab === 'demo' 
              ? 'bg-blue-50 border border-blue-200' 
              : 'bg-red-50 border border-red-200'
          }`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Shield className={`w-4 h-4 ${
                  activeTab === 'demo' ? 'text-blue-600' : 'text-red-600'
                }`} />
                <span className={`text-sm font-medium ${
                  activeTab === 'demo' ? 'text-blue-800' : 'text-red-800'
                }`}>
                  {activeTab === 'demo' ? 'Demo Account' : 'Real Account'}
                </span>
              </div>
              {hasCurrentToken && (
                <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded font-medium">
                  Configured
                </span>
              )}
            </div>
            <p className={`text-xs mt-1 ${
              activeTab === 'demo' ? 'text-blue-700' : 'text-red-700'
            }`}>
              {activeTab === 'demo' 
                ? 'Safe environment for testing strategies with virtual money'
                : 'Live trading with real money - use with caution'
              }
            </p>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-4 rounded-lg focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium ${
              activeTab === 'demo'
                ? 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500 text-white'
                : 'bg-red-600 hover:bg-red-700 focus:ring-red-500 text-white'
            }`}
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                {hasCurrentToken ? 'Updating' : 'Configuring'} {activeTab.toUpperCase()}...
              </div>
            ) : (
              `${hasCurrentToken ? 'Update' : 'Setup'} ${activeTab.toUpperCase()} API Token`
            )}
          </button>
        </form>

        {/* Help Link */}
        <div className="mt-4 text-center">
          <a
            href="https://app.deriv.com/account/api-token"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-blue-600 hover:text-blue-700"
          >
            How to get your API token?
          </a>
        </div>

        {/* Close Button */}
        <div className="mt-4 text-center">
          <button
            onClick={onClose}
            className="text-sm text-gray-600 hover:text-gray-800"
          >
            Close and use later
          </button>
        </div>
      </div>
    </div>
  );
};

export default ApiTokenSetup;
