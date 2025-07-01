"""
AI-powered trading analysis using OpenAI GPT
"""
import openai
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import config

logger = logging.getLogger(__name__)

class AIAnalyzer:
    def __init__(self):
        openai.api_key = config.OPENAI_API_KEY
        self.config = config.AI_CONFIG
        
    async def analyze_market_data(self, 
                                  market_data: Dict[str, Any], 
                                  technical_indicators: Dict[str, Any],
                                  historical_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze market data using AI and provide trading recommendation"""
        try:
            # Prepare context for AI analysis
            context = self._prepare_analysis_context(market_data, technical_indicators, historical_performance)
            
            # Generate AI analysis
            analysis = await self._get_ai_analysis(context)
            
            # Parse AI response
            result = self._parse_ai_response(analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {
                'prediction': 'NEUTRAL',
                'confidence': 0.5,
                'reasoning': 'Error in AI analysis',
                'risk_level': 'HIGH',
                'suggested_action': 'WAIT'
            }
            
    def _prepare_analysis_context(self, 
                                  market_data: Dict[str, Any], 
                                  technical_indicators: Dict[str, Any],
                                  historical_performance: Dict[str, Any] = None) -> str:
        """Prepare context string for AI analysis"""
        
        context = f"""
        MARKET DATA ANALYSIS REQUEST
        ===========================
        
        Current Market Data:
        - Symbol: {market_data.get('symbol', 'Unknown')}
        - Current Price: {market_data.get('current_price', 0)}
        - Price Change: {market_data.get('price_change', 0)}
        - Price Change %: {market_data.get('price_change_percent', 0)}
        - Volume: {market_data.get('volume', 0)}
        - Timestamp: {market_data.get('timestamp', datetime.now())}
        
        Technical Indicators:
        
        RSI Analysis:
        - RSI Value: {technical_indicators.get('rsi', {}).get('value', 50)}
        - RSI Signal: {technical_indicators.get('rsi', {}).get('signal', 'NEUTRAL')}
        - Overbought: {technical_indicators.get('rsi', {}).get('overbought', False)}
        - Oversold: {technical_indicators.get('rsi', {}).get('oversold', False)}
        
        MACD Analysis:
        - MACD: {technical_indicators.get('macd', {}).get('macd', 0)}
        - Signal Line: {technical_indicators.get('macd', {}).get('signal', 0)}
        - Histogram: {technical_indicators.get('macd', {}).get('histogram', 0)}
        - Signal Type: {technical_indicators.get('macd', {}).get('signal_type', 'NEUTRAL')}
        
        Bollinger Bands:
        - Upper Band: {technical_indicators.get('bollinger', {}).get('upper', 0)}
        - Middle Band: {technical_indicators.get('bollinger', {}).get('middle', 0)}
        - Lower Band: {technical_indicators.get('bollinger', {}).get('lower', 0)}
        - Position: {technical_indicators.get('bollinger', {}).get('position', 'MIDDLE')}
        
        Moving Averages:
        - Trend Signal: {technical_indicators.get('moving_averages', {}).get('trend_signal', 'NEUTRAL')}
        - Golden Cross: {technical_indicators.get('moving_averages', {}).get('golden_cross', False)}
        - Price Above MA20: {technical_indicators.get('moving_averages', {}).get('price_above_ma20', True)}
        
        Trend Analysis:
        - Direction: {technical_indicators.get('trend', {}).get('direction', 'NEUTRAL')}
        - Strength: {technical_indicators.get('trend', {}).get('strength', 25)}
        - Strong Trend: {technical_indicators.get('trend', {}).get('strong_trend', False)}
        - Trend Score: {technical_indicators.get('trend', {}).get('trend_score', 0.5)}
        
        Volatility:
        - ATR: {technical_indicators.get('volatility', {}).get('atr', 0)}
        - High Volatility: {technical_indicators.get('volatility', {}).get('high_volatility', False)}
        
        Support/Resistance:
        - Resistance Levels: {technical_indicators.get('support_resistance', {}).get('resistance', [])}
        - Support Levels: {technical_indicators.get('support_resistance', {}).get('support', [])}
        
        Candlestick Patterns:
        {json.dumps(technical_indicators.get('candlestick_patterns', {}), indent=2)}
        """
        
        if historical_performance:
            context += f"""
            
        Historical Performance:
        - Total Trades: {historical_performance.get('total_trades', 0)}
        - Win Rate: {historical_performance.get('win_rate', 0)}%
        - Average Profit: {historical_performance.get('average_profit', 0)}
        - Recent Performance: {historical_performance.get('recent_trend', 'Unknown')}
        """
        
        return context
        
    async def _get_ai_analysis(self, context: str) -> str:
        """Get AI analysis from OpenAI"""
        try:
            prompt = f"""
            You are an expert cryptocurrency and forex trading analyst with deep knowledge of technical analysis, market psychology, and risk management. 
            
            Analyze the following market data and technical indicators to provide a trading recommendation.
            
            {context}
            
            Based on this data, provide your analysis in the following JSON format:
            {{
                "prediction": "BUY/SELL/NEUTRAL",
                "confidence": 0.0-1.0,
                "reasoning": "Detailed explanation of your analysis",
                "key_factors": ["List of key factors influencing your decision"],
                "risk_level": "LOW/MEDIUM/HIGH",
                "suggested_action": "BUY/SELL/WAIT/REDUCE_POSITION",
                "entry_points": [list of suggested entry prices],
                "stop_loss": suggested_stop_loss_level,
                "take_profit": suggested_take_profit_level,
                "time_horizon": "SHORT/MEDIUM/LONG",
                "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
                "volatility_assessment": "Analysis of current volatility",
                "confluence_factors": "Areas where multiple indicators agree"
            }}
            
            Consider:
            1. Multiple timeframe analysis
            2. Risk-reward ratio
            3. Market context and trends
            4. Volatility and liquidity
            5. Support/resistance levels
            6. Momentum indicators
            7. Volume analysis
            8. Candlestick patterns
            9. Overall market sentiment
            
            Be conservative in your recommendations and always consider risk management.
            """
            
            response = await openai.ChatCompletion.acreate(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are an expert trading analyst providing objective, data-driven analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return self._get_fallback_analysis()
            
    def _get_fallback_analysis(self) -> str:
        """Provide fallback analysis when AI is unavailable"""
        return """
        {
            "prediction": "NEUTRAL",
            "confidence": 0.5,
            "reasoning": "AI analysis unavailable, using conservative approach",
            "key_factors": ["Technical indicators mixed", "Market uncertainty"],
            "risk_level": "HIGH",
            "suggested_action": "WAIT",
            "entry_points": [],
            "stop_loss": null,
            "take_profit": null,
            "time_horizon": "SHORT",
            "market_sentiment": "NEUTRAL",
            "volatility_assessment": "Unable to assess",
            "confluence_factors": "Insufficient data"
        }
        """
        
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response and extract structured data"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                parsed_response = json.loads(json_str)
                
                # Validate and clean response
                return self._validate_ai_response(parsed_response)
            else:
                logger.error("Could not find JSON in AI response")
                return self._get_default_response()
                
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing AI response JSON: {e}")
            return self._get_default_response()
        except Exception as e:
            logger.error(f"Error processing AI response: {e}")
            return self._get_default_response()
            
    def _validate_ai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean AI response"""
        # Ensure required fields exist with defaults
        validated = {
            'prediction': response.get('prediction', 'NEUTRAL').upper(),
            'confidence': max(0.0, min(1.0, float(response.get('confidence', 0.5)))),
            'reasoning': response.get('reasoning', 'No reasoning provided'),
            'key_factors': response.get('key_factors', []),
            'risk_level': response.get('risk_level', 'MEDIUM').upper(),
            'suggested_action': response.get('suggested_action', 'WAIT').upper(),
            'entry_points': response.get('entry_points', []),
            'stop_loss': response.get('stop_loss'),
            'take_profit': response.get('take_profit'),
            'time_horizon': response.get('time_horizon', 'SHORT').upper(),
            'market_sentiment': response.get('market_sentiment', 'NEUTRAL').upper(),
            'volatility_assessment': response.get('volatility_assessment', 'Unknown'),
            'confluence_factors': response.get('confluence_factors', 'None identified')
        }
        
        # Validate prediction values
        if validated['prediction'] not in ['BUY', 'SELL', 'NEUTRAL']:
            validated['prediction'] = 'NEUTRAL'
            
        # Validate risk level
        if validated['risk_level'] not in ['LOW', 'MEDIUM', 'HIGH']:
            validated['risk_level'] = 'MEDIUM'
            
        # Validate suggested action
        if validated['suggested_action'] not in ['BUY', 'SELL', 'WAIT', 'REDUCE_POSITION']:
            validated['suggested_action'] = 'WAIT'
            
        return validated
        
    def _get_default_response(self) -> Dict[str, Any]:
        """Get default response when AI analysis fails"""
        return {
            'prediction': 'NEUTRAL',
            'confidence': 0.5,
            'reasoning': 'AI analysis failed, using conservative approach',
            'key_factors': ['Analysis error'],
            'risk_level': 'HIGH',
            'suggested_action': 'WAIT',
            'entry_points': [],
            'stop_loss': None,
            'take_profit': None,
            'time_horizon': 'SHORT',
            'market_sentiment': 'NEUTRAL',
            'volatility_assessment': 'Unable to assess',
            'confluence_factors': 'Analysis unavailable'
        }
        
    async def generate_strategy_analysis(self, 
                                         symbol: str, 
                                         timeframe: str, 
                                         market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive strategy analysis"""
        try:
            strategy_prompt = f"""
            Analyze the market data for {symbol} on {timeframe} timeframe and suggest trading strategies.
            
            Market Data Points: {len(market_data)}
            Recent Price Action: {json.dumps(market_data[-10:], indent=2) if market_data else 'No data'}
            
            Provide a comprehensive strategy analysis including:
            1. Current market phase (accumulation, markup, distribution, decline)
            2. Optimal entry strategies
            3. Risk management approach
            4. Position sizing recommendations
            5. Market timing considerations
            6. Alternative scenarios and contingency plans
            
            Format your response as a structured analysis with clear sections.
            """
            
            response = await openai.ChatCompletion.acreate(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are a professional trading strategist with expertise in market cycle analysis and risk management."},
                    {"role": "user", "content": strategy_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            return {
                'strategy_analysis': response.choices[0].message.content,
                'generated_at': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy analysis: {e}")
            return {
                'strategy_analysis': 'Strategy analysis unavailable due to technical error.',
                'generated_at': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
    async def evaluate_trade_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate trading performance and provide insights"""
        try:
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('profit_loss', 0) > 0])
            losing_trades = total_trades - winning_trades
            
            total_profit = sum(t.get('profit_loss', 0) for t in trades)
            avg_profit = total_profit / total_trades if total_trades > 0 else 0
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            performance_prompt = f"""
            Analyze the following trading performance data and provide insights:
            
            Performance Metrics:
            - Total Trades: {total_trades}
            - Winning Trades: {winning_trades}
            - Losing Trades: {losing_trades}
            - Win Rate: {win_rate:.2f}%
            - Total Profit/Loss: {total_profit:.2f}
            - Average Profit per Trade: {avg_profit:.2f}
            
            Recent Trades:
            {json.dumps(trades[-20:], indent=2) if trades else 'No recent trades'}
            
            Provide analysis on:
            1. Performance strengths and weaknesses
            2. Pattern recognition in wins/losses
            3. Risk management effectiveness
            4. Suggestions for improvement
            5. Market condition correlation
            6. Recommended adjustments to strategy
            """
            
            response = await openai.ChatCompletion.acreate(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "You are a trading performance analyst specializing in strategy optimization and risk management."},
                    {"role": "user", "content": performance_prompt}
                ],
                temperature=0.1,
                max_tokens=1200
            )
            
            return {
                'performance_analysis': response.choices[0].message.content,
                'metrics': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit
                },
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating trade performance: {e}")
            return {
                'performance_analysis': 'Performance analysis unavailable due to technical error.',
                'metrics': {'total_trades': 0, 'win_rate': 0, 'total_profit': 0, 'avg_profit': 0},
                'generated_at': datetime.now().isoformat()
            }
