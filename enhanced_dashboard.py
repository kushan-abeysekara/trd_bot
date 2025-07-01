"""
Enhanced Deriv AI Trading Bot Dashboard with Real-time Features
Professional-grade dashboard with live controls, analytics, and monitoring
"""
import asyncio
import json
import subprocess
import psutil
import os
import sys
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func, and_
from database import SessionLocal, Trade, TradingSession, AIAnalysis
import config
import logging
from deriv_api import DerivAPI
from ai_analyzer import AIAnalyzer
from technical_analysis import TechnicalAnalyzer
import random
import websockets
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Deriv AI Trading Bot Dashboard", 
    version="3.0.0",
    description="Professional trading dashboard with real-time controls and analytics"
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients with error handling"""
        if not self.active_connections:
            return
            
        # Add timestamp to all broadcasts
        message["timestamp"] = datetime.now().isoformat()
        message_str = json.dumps(message)
        
        # Keep track of failed connections to remove them
        failed_connections = []
        
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send WebSocket message: {e}")
                failed_connections.append(connection)
        
        # Remove failed connections
        for failed_conn in failed_connections:
            self.disconnect(failed_conn)
    
    async def send_to_client(self, websocket: WebSocket, message: dict):
        """Send message to specific client with error handling"""
        try:
            message["timestamp"] = datetime.now().isoformat()
            await websocket.send_text(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to client: {e}")
            self.disconnect(websocket)
            return False

manager = ConnectionManager()

# Global bot control state and process management
bot_process = None
bot_state = {
    "is_running": False,
    "trading_enabled": False,
    "account_balance": 10000.0,
    "current_trades": [],
    "session_profit": 0.0,
    "total_trades_today": 0,
    "win_rate": 0.0,
    "last_analysis": None,
    "last_10_digits": [],
    "connection_status": "Connecting...",
    "last_tick": None,
    "market_data": {},
    "ai_analysis_running": False,
    "last_update": datetime.now(),
    "uptime": 0,
    "equity": 10000.0,
    "free_margin": 10000.0,
    "margin": 0.0,
    "real_time_price": 0.0,
    "price_history": [],
    "digits_stream": [],
    "connection_stable": False,
    "reconnect_count": 0,
    "last_ai_analysis": {},
    "market_trend": "NEUTRAL",
    "volatility": 0.0
}

# Initialize AI components and real-time connections
deriv_api = DerivAPI()
ai_analyzer = AIAnalyzer()
technical_analyzer = TechnicalAnalyzer()

# Real-time market data connection
market_websocket = None
analysis_task = None
realtime_analysis_task = None
connection_monitor_task = None

# Connection state management - SIMPLIFIED
connection_state = {
    "is_stable": False,
    "last_successful_connection": None,
    "reconnect_attempts": 0,
    "max_reconnect_attempts": 3,  # Reduced attempts
    "reconnect_delay": 30  # Increased delay to 30 seconds
}

# Background task to update bot state
async def update_bot_state_periodically():
    """Update bot state every 10 seconds - reduced frequency"""
    global bot_state
    while True:
        try:
            # Update from database
            await update_bot_state_from_db()
            
            # Update connection status
            bot_state["connection_stable"] = connection_state["is_stable"]
            bot_state["reconnect_count"] = connection_state["reconnect_attempts"]
            bot_state["connection_status"] = "Connected" if connection_state["is_stable"] else "Connecting..."
            
            # Broadcast updates to all connected clients (less frequently)
            await manager.broadcast({
                "type": "bot_status_update",
                "data": bot_state
            })
            
        except Exception as e:
            logger.error(f"Error updating bot state: {e}")
            
        await asyncio.sleep(10)  # Increased from 5 to 10 seconds

async def update_bot_state_from_db():
    """Update bot state from database"""
    global bot_state
    try:
        db = SessionLocal()
        try:
            # Get today's trades
            today = datetime.now().date()
            today_trades = db.query(Trade).filter(
                func.date(Trade.start_time) == today,
                Trade.status.in_(['WON', 'LOST'])
            ).all()
            
            # Calculate statistics
            total_trades = len(today_trades)
            won_trades = len([t for t in today_trades if t.status == 'WON'])
            session_profit = sum([t.profit_loss or 0 for t in today_trades])
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Get current active trades
            active_trades = db.query(Trade).filter(Trade.status == 'PENDING').all()
            
            # Update bot state
            bot_state.update({
                "total_trades_today": total_trades,
                "win_rate": round(win_rate, 1),
                "session_profit": round(session_profit, 2),
                "current_trades": len(active_trades),
                "last_update": datetime.now(),
                "equity": bot_state["account_balance"] + session_profit,
                "free_margin": bot_state["account_balance"] - (len(active_trades) * 10)  # Estimate
            })
            
            # Check if bot process is running
            if bot_process and bot_process.poll() is None:
                bot_state["is_running"] = True
                bot_state["trading_enabled"] = True
            else:
                bot_state["is_running"] = False
                bot_state["trading_enabled"] = False
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error updating bot state from DB: {e}")

@app.on_event("startup")
async def startup_event():
    """Start background tasks on app startup"""
    global realtime_analysis_task, connection_monitor_task
    
    logger.info("🚀 Starting Enhanced Dashboard...")
    
    # Start periodic bot state updates
    asyncio.create_task(update_bot_state_periodically())
    
    # Start connection monitor
    connection_monitor_task = asyncio.create_task(monitor_connection_health())
    
    # Start real-time analysis if not already running
    if config.DEMO_MODE:
        # Use simulation for demo mode
        realtime_analysis_task = asyncio.create_task(simulate_market_data())
    else:
        # Use real API connection
        realtime_analysis_task = asyncio.create_task(run_realtime_ai_analysis())
    
    logger.info("✅ Dashboard background tasks started")

@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Modern Professional Dashboard Home Page"""
    return templates.TemplateResponse("modern_dashboard.html", {
        "request": request,
        "bot_state": bot_state
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates with improved error handling"""
    client_id = f"client_{datetime.now().timestamp()}"
    logger.info(f"🔌 New WebSocket connection: {client_id}")
    
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "data": bot_state
        }))
        
        # Keep connection alive with periodic updates
        while True:
            try:
                # Wait for any incoming messages (ping/pong)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=2.0)
                
                # Handle client messages
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}: {message}")
                
            except asyncio.TimeoutError:
                # No message received, send status update
                try:
                    await websocket.send_text(json.dumps({
                        "type": "bot_status",
                        "timestamp": datetime.now().isoformat(),
                        "data": bot_state
                    }))
                except Exception as e:
                    logger.warning(f"Failed to send status update to {client_id}: {e}")
                    break
            
            except Exception as e:
                logger.warning(f"Error handling message from {client_id}: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"🔌 WebSocket error for {client_id}: {e}")
    finally:
        manager.disconnect(websocket)
        logger.info(f"🔌 WebSocket connection {client_id} closed")

@app.post("/api/bot/start")
async def start_bot(background_tasks: BackgroundTasks):
    """Start the trading bot"""
    global bot_process, bot_state
    
    try:
        # Check if bot is already running
        if bot_process and bot_process.poll() is None:
            return {"status": "info", "message": "Bot is already running"}
        
        # Start the main bot process
        logger.info("Starting trading bot process...")
        bot_process = subprocess.Popen([
            "python", "main.py"
        ], cwd=os.getcwd())
        
        # Update state
        bot_state["is_running"] = True
        bot_state["trading_enabled"] = True
        bot_state["last_update"] = datetime.now()
        
        # Broadcast update
        await manager.broadcast({
            "type": "bot_started",
            "message": "Trading bot started successfully",
            "timestamp": datetime.now().isoformat(),
            "data": bot_state
        })
        
        logger.info(f"Bot process started with PID: {bot_process.pid}")
        return {"status": "success", "message": "Bot started successfully", "pid": bot_process.pid}
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        bot_state["is_running"] = False
        bot_state["trading_enabled"] = False
        return {"status": "error", "message": f"Failed to start bot: {str(e)}"}

@app.post("/api/bot/stop")
async def stop_bot():
    """Stop the trading bot"""
    global bot_process, bot_state
    
    try:
        if not bot_process or bot_process.poll() is not None:
            return {"status": "info", "message": "Bot is not running"}
        
        # Terminate the bot process
        logger.info(f"Stopping bot process (PID: {bot_process.pid})...")
        bot_process.terminate()
        
        # Wait for process to end (with timeout)
        try:
            bot_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Bot process didn't terminate gracefully, killing it...")
            bot_process.kill()
            bot_process.wait()
        
        # Update state
        bot_state["is_running"] = False
        bot_state["trading_enabled"] = False
        bot_state["last_update"] = datetime.now()
        
        # Broadcast update
        await manager.broadcast({
            "type": "bot_stopped",
            "message": "Trading bot stopped successfully",
            "timestamp": datetime.now().isoformat(),
            "data": bot_state
        })
        
        logger.info("Bot process stopped successfully")
        return {"status": "success", "message": "Bot stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return {"status": "error", "message": f"Failed to stop bot: {str(e)}"}

@app.get("/api/bot/status")
async def get_bot_status():
    """Get comprehensive bot status"""
    global bot_process, bot_state
    
    # Check if process is actually running
    if bot_process:
        is_process_running = bot_process.poll() is None
        bot_state["is_running"] = is_process_running
        bot_state["trading_enabled"] = is_process_running
    else:
        bot_state["is_running"] = False
        bot_state["trading_enabled"] = False
    
    # Add system info
    status_data = {
        **bot_state,
        "process_id": bot_process.pid if bot_process and bot_process.poll() is None else None,
        "system_time": datetime.now().isoformat(),
        "uptime_minutes": (datetime.now() - bot_state["last_update"]).total_seconds() / 60 if bot_state["is_running"] else 0
    }
    
    return status_data

@app.get("/api/account/balance")
async def get_account_balance():
    """Get current account balance with detailed information"""
    try:
        # Get real-time data from database
        db = SessionLocal()
        try:
            # Calculate current session P&L
            today = datetime.now().date()
            today_trades = db.query(Trade).filter(
                func.date(Trade.start_time) == today,
                Trade.status.in_(['WON', 'LOST'])
            ).all()
            
            session_pnl = sum([t.profit_loss or 0 for t in today_trades])
            
            # Get pending trades (active positions)
            pending_trades = db.query(Trade).filter(Trade.status == 'PENDING').all()
            total_margin = sum([t.stake or 0 for t in pending_trades])
            
            balance_data = {
                "balance": bot_state["account_balance"],
                "currency": "USD",
                "equity": bot_state["account_balance"] + session_pnl,
                "session_pnl": round(session_pnl, 2),
                "margin_used": round(total_margin, 2),
                "free_margin": round(bot_state["account_balance"] - total_margin, 2),
                "active_positions": len(pending_trades),
                "last_update": datetime.now().isoformat()
            }
            
            # Update global state
            bot_state.update({
                "session_profit": round(session_pnl, 2),
                "equity": round(bot_state["account_balance"] + session_pnl, 2),
                "margin": round(total_margin, 2),
                "free_margin": round(bot_state["account_balance"] - total_margin, 2)
            })
            
            return balance_data
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        # Return fallback data
        return {
            "balance": bot_state["account_balance"],
            "currency": "USD",
            "equity": bot_state["account_balance"],
            "session_pnl": bot_state["session_profit"],
            "margin_used": 0.0,
            "free_margin": bot_state["account_balance"],
            "active_positions": 0,
            "last_update": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/trades/current")
async def get_current_trades():
    """Get current active trades with real-time data"""
    try:
        db = SessionLocal()
        try:
            active_trades = db.query(Trade).filter(
                Trade.status == 'PENDING'
            ).order_by(desc(Trade.start_time)).limit(20).all()
            
            trades_data = []
            for trade in active_trades:
                # Calculate time elapsed
                time_elapsed = 0
                if trade.start_time:
                    time_elapsed = (datetime.now() - trade.start_time).total_seconds()
                
                # Estimate current P&L (simplified)
                current_pnl = 0.0
                if trade.duration and time_elapsed > 0:
                    # Simple estimation based on time progress
                    progress = min(time_elapsed / trade.duration, 1.0)
                    # This is a simplified calculation - in reality you'd get live prices
                    current_pnl = trade.stake * 0.85 * progress if trade.trade_type == 'CALL' else -trade.stake * 0.1
                
                trades_data.append({
                    "id": trade.id,
                    "contract_id": trade.contract_id,
                    "symbol": trade.symbol,
                    "type": trade.trade_type,
                    "stake": round(trade.stake, 2),
                    "entry_time": trade.start_time.isoformat() if trade.start_time else None,
                    "duration": trade.duration,
                    "time_elapsed": round(time_elapsed, 1),
                    "time_remaining": max(0, trade.duration - time_elapsed) if trade.duration else 0,
                    "status": trade.status,
                    "ai_confidence": round(trade.ai_confidence * 100, 1) if trade.ai_confidence else 0,
                    "current_pnl": round(current_pnl, 2),
                    "entry_price": trade.entry_price,
                    "martingale_level": trade.martingale_level or 0
                })
            
            return {"current_trades": trades_data, "total_active": len(trades_data)}
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting current trades: {e}")
        return {"current_trades": [], "total_active": 0, "error": str(e)}

@app.get("/api/trades/history")
async def get_trade_history(limit: int = 50):
    """Get recent trade history with detailed information"""
    try:
        db = SessionLocal()
        try:
            recent_trades = db.query(Trade).filter(
                Trade.status.in_(['WON', 'LOST'])
            ).order_by(desc(Trade.end_time)).limit(limit).all()
            
            trades_data = []
            total_profit = 0
            total_loss = 0
            
            for trade in recent_trades:
                profit_loss = trade.profit_loss or 0
                if profit_loss > 0:
                    total_profit += profit_loss
                else:
                    total_loss += abs(profit_loss)
                
                # Calculate trade duration in seconds
                duration_seconds = 0
                if trade.start_time and trade.end_time:
                    duration_seconds = (trade.end_time - trade.start_time).total_seconds()
                
                trades_data.append({
                    "id": trade.id,
                    "contract_id": trade.contract_id,
                    "symbol": trade.symbol,
                    "type": trade.trade_type,
                    "stake": round(trade.stake, 2),
                    "profit_loss": round(profit_loss, 2),
                    "status": trade.status,
                    "start_time": trade.start_time.isoformat() if trade.start_time else None,
                    "end_time": trade.end_time.isoformat() if trade.end_time else None,
                    "duration": trade.duration,
                    "actual_duration": round(duration_seconds, 1),
                    "ai_confidence": round(trade.ai_confidence * 100, 1) if trade.ai_confidence else 0,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "martingale_level": trade.martingale_level or 0,
                    "payout_ratio": round((trade.stake + profit_loss) / trade.stake, 2) if trade.stake > 0 else 0
                })
            
            # Calculate summary statistics
            won_trades = len([t for t in recent_trades if t.status == 'WON'])
            total_trades = len(recent_trades)
            win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
            
            return {
                "trade_history": trades_data,
                "summary": {
                    "total_trades": total_trades,
                    "won_trades": won_trades,
                    "lost_trades": total_trades - won_trades,
                    "win_rate": round(win_rate, 1),
                    "total_profit": round(total_profit, 2),
                    "total_loss": round(total_loss, 2),
                    "net_profit": round(total_profit - total_loss, 2)
                }
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        return {
            "trade_history": [],
            "summary": {
                "total_trades": 0,
                "won_trades": 0,
                "lost_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "total_loss": 0,
                "net_profit": 0
            },
            "error": str(e)
        }

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get comprehensive performance analytics and charts data"""
    try:
        db = SessionLocal()
        try:
            # Get performance data for different time periods
            now = datetime.now()
            periods = {
                "today": now.date(),
                "yesterday": (now - timedelta(days=1)).date(),
                "last_7_days": now - timedelta(days=7),
                "last_30_days": now - timedelta(days=30)
            }
            
            # Daily performance for last 30 days
            daily_performance = db.query(
                func.date(Trade.start_time).label('date'),
                func.sum(Trade.profit_loss).label('daily_profit'),
                func.count(Trade.id).label('trades_count'),
                func.sum(func.case([(Trade.status == 'WON', 1)], else_=0)).label('wins'),
                func.sum(Trade.stake).label('total_volume')
            ).filter(
                Trade.start_time >= periods["last_30_days"],
                Trade.status.in_(['WON', 'LOST'])
            ).group_by(func.date(Trade.start_time)).order_by(func.date(Trade.start_time)).all()
            
            # Overall statistics
            all_trades = db.query(Trade).filter(Trade.status.in_(['WON', 'LOST'])).all()
            total_trades = len(all_trades)
            winning_trades = len([t for t in all_trades if t.status == 'WON'])
            total_profit = sum([t.profit_loss or 0 for t in all_trades])
            total_volume = sum([t.stake or 0 for t in all_trades])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
            
            # Hourly performance (for today)
            today_start = datetime.combine(now.date(), datetime.min.time())
            hourly_performance = db.query(
                func.hour(Trade.start_time).label('hour'),
                func.sum(Trade.profit_loss).label('hourly_profit'),
                func.count(Trade.id).label('trades_count')
            ).filter(
                Trade.start_time >= today_start,
                Trade.status.in_(['WON', 'LOST'])
            ).group_by(func.hour(Trade.start_time)).all()
            
            # Symbol performance
            symbol_performance = db.query(
                Trade.symbol,
                func.count(Trade.id).label('trades_count'),
                func.sum(Trade.profit_loss).label('total_profit'),
                func.sum(func.case([(Trade.status == 'WON', 1)], else_=0)).label('wins')
            ).filter(
                Trade.status.in_(['WON', 'LOST'])
            ).group_by(Trade.symbol).all()
            
            # Format data for charts
            chart_data = {
                "daily_performance": [
                    {
                        "date": row.date.strftime("%Y-%m-%d"),
                        "profit": float(row.daily_profit or 0),
                        "trades": int(row.trades_count),
                        "wins": int(row.wins),
                        "win_rate": round((row.wins / row.trades_count * 100), 1) if row.trades_count > 0 else 0,
                        "volume": float(row.total_volume or 0)
                    } for row in daily_performance
                ],
                "hourly_performance": [
                    {
                        "hour": int(row.hour),
                        "profit": float(row.hourly_profit or 0),
                        "trades": int(row.trades_count)
                    } for row in hourly_performance
                ],
                "symbol_performance": [
                    {
                        "symbol": row.symbol,
                        "trades": int(row.trades_count),
                        "profit": float(row.total_profit or 0),
                        "wins": int(row.wins),
                        "win_rate": round((row.wins / row.trades_count * 100), 1) if row.trades_count > 0 else 0
                    } for row in symbol_performance
                ],
                "overall_statistics": {
                    "total_trades": total_trades,
                    "winning_trades": winning_trades,
                    "losing_trades": total_trades - winning_trades,
                    "win_rate": round(win_rate, 2),
                    "total_profit": round(float(total_profit), 2),
                    "total_volume": round(float(total_volume), 2),
                    "average_profit_per_trade": round(float(avg_profit_per_trade), 2),
                    "profit_factor": round(abs(total_profit / min(abs(sum([t.profit_loss for t in all_trades if (t.profit_loss or 0) < 0])), 1)), 2),
                    "sharpe_ratio": 0.0,  # Would need more complex calculation
                    "max_drawdown": 0.0   # Would need more complex calculation
                },
                "last_updated": datetime.now().isoformat()
            }
            
            return chart_data
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        # Return empty structure on error
        return {
            "daily_performance": [],
            "hourly_performance": [],
            "symbol_performance": [],
            "overall_statistics": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_profit": 0,
                "total_volume": 0,
                "average_profit_per_trade": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0
            },
            "last_updated": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/analysis/latest")
async def get_latest_analysis():
    """Get latest AI analysis with comprehensive data"""
    try:
        db = SessionLocal()
        try:
            # Get the most recent analysis
            latest_analysis = db.query(AIAnalysis).order_by(desc(AIAnalysis.timestamp)).first()
            
            if latest_analysis:
                # Parse technical data if it's JSON string
                technical_data = {}
                try:
                    if latest_analysis.technical_data:
                        if isinstance(latest_analysis.technical_data, str):
                            technical_data = json.loads(latest_analysis.technical_data)
                        else:
                            technical_data = latest_analysis.technical_data
                except (json.JSONDecodeError, TypeError):
                    technical_data = {}
                
                analysis_data = {
                    "timestamp": latest_analysis.timestamp.isoformat(),
                    "symbol": latest_analysis.symbol,
                    "prediction": latest_analysis.prediction,
                    "confidence": round(latest_analysis.confidence * 100, 1),
                    "analysis_text": latest_analysis.analysis_text,
                    "technical_data": technical_data,
                    "market_conditions": {
                        "trend": technical_data.get("trend", "NEUTRAL"),
                        "volatility": technical_data.get("volatility", {}).get("level", "MEDIUM"),
                        "support_level": technical_data.get("support_resistance", {}).get("support", 0),
                        "resistance_level": technical_data.get("support_resistance", {}).get("resistance", 0),
                        "rsi": technical_data.get("rsi", {}).get("current", 50),
                        "macd_signal": technical_data.get("macd", {}).get("signal", "NEUTRAL")
                    },
                    "risk_assessment": {
                        "level": "MEDIUM",
                        "factors": ["Market volatility", "Technical indicators alignment"],
                        "recommended_stake": 10.0
                    },
                    "age_minutes": (datetime.now() - latest_analysis.timestamp).total_seconds() / 60
                }
            else:
                # Generate sample analysis if none exists
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "symbol": "R_10",
                    "prediction": "NEUTRAL",
                    "confidence": 50.0,
                    "analysis_text": "No recent analysis available. Start the trading bot to generate AI market analysis.",
                    "technical_data": {},
                    "market_conditions": {
                        "trend": "NEUTRAL",
                        "volatility": "MEDIUM",
                        "support_level": 0,
                        "resistance_level": 0,
                        "rsi": 50,
                        "macd_signal": "NEUTRAL"
                    },
                    "risk_assessment": {
                        "level": "LOW",
                        "factors": ["No recent data"],
                        "recommended_stake": 10.0
                    },
                    "age_minutes": 0
                }
            
            return analysis_data
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting latest analysis: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": "R_10",
            "prediction": "ERROR",
            "confidence": 0.0,
            "analysis_text": f"Error retrieving analysis: {str(e)}",
            "technical_data": {},
            "market_conditions": {
                "trend": "UNKNOWN",
                "volatility": "UNKNOWN",
                "support_level": 0,
                "resistance_level": 0,
                "rsi": 0,
                "macd_signal": "UNKNOWN"
            },
            "risk_assessment": {
                "level": "HIGH",
                "factors": ["System error"],
                "recommended_stake": 0
            },
            "age_minutes": 999,
            "error": str(e)
        }

# Real-time AI Analysis
async def run_realtime_ai_analysis():
    """Run real-time AI market analysis with improved connection handling"""
    global bot_state
    
    bot_state["ai_analysis_running"] = True
    logger.info("🧠 Starting real-time AI analysis...")
    
    retry_count = 0
    max_retries = 5
    
    while bot_state["ai_analysis_running"] and retry_count < max_retries:
        try:
            # Ensure we have a stable connection
            if not await ensure_stable_connection():
                logger.error("❌ Could not establish stable connection for AI analysis")
                retry_count += 1
                await asyncio.sleep(10)
                continue
            
            logger.info("✅ Connected to Deriv API for real-time analysis")
            
            # Reset retry count on successful connection
            retry_count = 0
            
            while bot_state["ai_analysis_running"] and bot_state["is_running"] and deriv_api.is_connected:
                try:
                    # Get market data for multiple symbols
                    symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
                    
                    for symbol in symbols:
                        try:
                            # Get current tick data with timeout
                            tick_data = await asyncio.wait_for(
                                deriv_api.get_ticks(symbol), 
                                timeout=5.0
                            )
                            
                            if tick_data and 'tick' in tick_data:
                                tick = tick_data['tick']
                                current_price = float(tick['quote'])
                                
                                # Update market data
                                if symbol not in bot_state["market_data"]:
                                    bot_state["market_data"][symbol] = {
                                        "last_10_digits": [],
                                        "current_price": current_price,
                                        "last_update": datetime.now().isoformat(),
                                        "price_history": []
                                    }
                                
                                # Extract last digit and add to array
                                last_digit = int(str(current_price).replace('.', '')[-1])
                                bot_state["market_data"][symbol]["last_10_digits"].append(last_digit)
                                bot_state["market_data"][symbol]["current_price"] = current_price
                                bot_state["market_data"][symbol]["last_update"] = datetime.now().isoformat()
                                
                                # Add to price history
                                bot_state["market_data"][symbol]["price_history"].append({
                                    "price": current_price,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                # Keep only last 10 digits and 100 price points
                                if len(bot_state["market_data"][symbol]["last_10_digits"]) > 10:
                                    bot_state["market_data"][symbol]["last_10_digits"].pop(0)
                                    
                                if len(bot_state["market_data"][symbol]["price_history"]) > 100:
                                    bot_state["market_data"][symbol]["price_history"].pop(0)
                                
                                # Update global last_10_digits for main symbol (R_10)
                                if symbol == "R_10":
                                    bot_state["last_10_digits"] = bot_state["market_data"][symbol]["last_10_digits"].copy()
                                    bot_state["real_time_price"] = current_price
                                    bot_state["last_tick"] = {
                                        "symbol": symbol,
                                        "price": current_price,
                                        "timestamp": datetime.now().isoformat()
                                    }
                                
                        except asyncio.TimeoutError:
                            logger.warning(f"⏱️ Timeout getting tick data for {symbol}")
                            continue
                        except Exception as e:
                            logger.warning(f"⚠️ Error getting tick data for {symbol}: {e}")
                            continue
                    
                    # Perform AI analysis every 30 seconds
                    if len(bot_state["last_10_digits"]) >= 5:
                        try:
                            analysis = await get_ai_analysis_safe(bot_state["last_10_digits"])
                            bot_state["last_ai_analysis"] = analysis
                            bot_state["market_trend"] = analysis.get("prediction", "NEUTRAL")
                            bot_state["volatility"] = analysis.get("confidence", 0.0)
                            
                            # Broadcast analysis update
                            await manager.broadcast({
                                "type": "ai_analysis_update",
                                "data": {
                                    "analysis": analysis,
                                    "market_data": bot_state["market_data"],
                                    "last_10_digits": bot_state["last_10_digits"]
                                }
                            })
                            
                        except Exception as e:
                            logger.error(f"❌ AI analysis error: {e}")
                    
                    # Update bot state timestamp
                    bot_state["last_update"] = datetime.now()
                    
                    # Broadcast market data update
                    await manager.broadcast({
                        "type": "market_data_update",
                        "data": {
                            "market_data": bot_state["market_data"],
                            "last_tick": bot_state["last_tick"],
                            "connection_stable": deriv_api.is_connected
                        }
                    })
                    
                    # Wait before next update
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Error in analysis loop: {e}")
                    if not deriv_api.is_connected:
                        logger.warning("Connection lost during analysis, will retry...")
                        break
                    await asyncio.sleep(5)
            
            # Connection lost, increment retry count
            if not deriv_api.is_connected:
                retry_count += 1
                logger.warning(f"Connection lost, retry {retry_count}/{max_retries}")
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"❌ Critical error in real-time analysis: {e}")
            retry_count += 1
            await asyncio.sleep(10)
    
    logger.warning("🔴 Real-time AI analysis stopped")
    bot_state["ai_analysis_running"] = False

async def get_ai_analysis_safe(digits_array):
    """Safely get AI analysis with error handling"""
    try:
        if not digits_array or len(digits_array) < 5:
            return {
                "prediction": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": "Insufficient data for analysis"
            }
        
        # Use the AI analyzer
        mock_market_data = {
            'current_price': digits_array[-1] if digits_array else 0,
            'timestamp': datetime.now(),
            'digits': digits_array
        }
        
        analysis = await ai_analyzer.analyze_market_data(
            mock_market_data, 
            {},  # technical indicators
            {}   # historical performance
        )
        
        return {
            "prediction": analysis.get("prediction", "NEUTRAL"),
            "confidence": analysis.get("confidence", 0.0),
            "reasoning": analysis.get("reasoning", "AI analysis completed")
        }
        
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        return {
            "prediction": "NEUTRAL",
            "confidence": 0.0,
            "reasoning": f"Analysis error: {str(e)}"
        }

async def simulate_market_data():
    """Simulate market data when API is not available"""
    global bot_state
    
    symbols = ["R_10", "R_25", "R_50", "R_75", "R_100"]
    base_prices = {"R_10": 100.0, "R_25": 250.0, "R_50": 500.0, "R_75": 750.0, "R_100": 1000.0}
    
    while bot_state["ai_analysis_running"]:
        try:
            for symbol in symbols:
                if symbol not in bot_state["market_data"]:
                    bot_state["market_data"][symbol] = {
                        "last_10_digits": [],
                        "current_price": base_prices[symbol],
                        "last_update": datetime.now().isoformat()
                    }
                
                # Simulate price movement
                current_price = bot_state["market_data"][symbol]["current_price"]
                price_change = random.uniform(-0.1, 0.1)
                new_price = current_price + price_change
                
                # Extract last digit
                last_digit = int(str(f"{new_price:.5f}").replace('.', '')[-1])
                bot_state["market_data"][symbol]["last_10_digits"].append(last_digit)
                
                # Keep only last 10 digits
                if len(bot_state["market_data"][symbol]["last_10_digits"]) > 10:
                    bot_state["market_data"][symbol]["last_10_digits"].pop(0)
                
                bot_state["market_data"][symbol]["current_price"] = new_price
                bot_state["market_data"][symbol]["last_update"] = datetime.now().isoformat()
            
            # Simulate AI analysis
            predictions = ["BUY", "SELL", "NEUTRAL"]
            trends = ["UPTREND", "DOWNTREND", "SIDEWAYS"]
            
            bot_state["last_analysis"] = {
                "timestamp": datetime.now().isoformat(),
                "symbol": "R_10",
                "prediction": random.choice(predictions),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "analysis_text": f"Market analysis at {datetime.now().strftime('%H:%M:%S')}",
                "market_trend": random.choice(trends),
                "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"]),
                "recommended_action": random.choice(["BUY", "SELL", "WAIT"])
            }
            
            # Broadcast update
            await manager.broadcast({
                "type": "realtime_analysis",
                "data": bot_state["last_analysis"],
                "market_data": bot_state["market_data"],
                "timestamp": datetime.now().isoformat()
            })
            
            await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error in market simulation: {e}")
            await asyncio.sleep(5)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send comprehensive real-time updates
            update_data = {
                "type": "dashboard_update",
                "timestamp": datetime.now().isoformat(),
                "bot_status": bot_state,
                "account_info": await get_account_balance(),
                "active_trades_count": len((await get_current_trades())["current_trades"]),
                "system_health": {
                    "dashboard_uptime": "Online",
                    "last_data_update": bot_state["last_update"].isoformat(),
                    "connections": len(manager.active_connections)
                }
            }
            
            await websocket.send_text(json.dumps(update_data))
            await asyncio.sleep(3)  # Update every 3 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Connection monitoring and stability
async def monitor_connection_health():
    """Monitor connection health with reduced frequency"""
    global connection_state, bot_state
    
    logger.info("🔍 Starting connection health monitor...")
    
    while True:
        try:
            # Check Deriv API connection less frequently
            if deriv_api.is_connected:
                if not connection_state["is_stable"]:
                    logger.info("✅ Connection restored")
                    
                connection_state["is_stable"] = True
                connection_state["last_successful_connection"] = datetime.now()
                connection_state["reconnect_attempts"] = 0
                bot_state["connection_status"] = "Connected"
                bot_state["connection_stable"] = True
            else:
                connection_state["is_stable"] = False
                bot_state["connection_status"] = "Disconnected"
                bot_state["connection_stable"] = False
                
                # Only attempt reconnection if not at max attempts and enough time has passed
                if (connection_state["reconnect_attempts"] < connection_state["max_reconnect_attempts"] and
                    (connection_state["last_successful_connection"] is None or 
                     (datetime.now() - connection_state["last_successful_connection"]).total_seconds() > 60)):
                    
                    logger.warning(f"Connection lost, attempting reconnection {connection_state['reconnect_attempts'] + 1}/{connection_state['max_reconnect_attempts']}")
                    
                    connection_state["reconnect_attempts"] += 1
                    bot_state["reconnect_count"] = connection_state["reconnect_attempts"]
                    
                    try:
                        # Try to reconnect with extended delay
                        await deriv_api.connect()
                        if deriv_api.is_connected:
                            logger.info("✅ Reconnection successful!")
                            connection_state["is_stable"] = True
                            connection_state["reconnect_attempts"] = 0
                            bot_state["connection_status"] = "Connected"
                            bot_state["connection_stable"] = True
                        else:
                            logger.warning("❌ Reconnection failed")
                            
                    except Exception as e:
                        logger.error(f"Reconnection error: {e}")
                
                elif connection_state["reconnect_attempts"] >= connection_state["max_reconnect_attempts"]:
                    # Reset after extended wait
                    if (connection_state["last_successful_connection"] is None or 
                        (datetime.now() - connection_state["last_successful_connection"]).total_seconds() > 300):
                        logger.info("Resetting reconnection attempts after extended wait")
                        connection_state["reconnect_attempts"] = 0
            
            # Broadcast connection status update less frequently
            await manager.broadcast({
                "type": "connection_status",
                "data": {
                    "connection_stable": connection_state["is_stable"],
                    "reconnect_count": connection_state["reconnect_attempts"],
                    "status": bot_state["connection_status"],
                    "last_successful": connection_state["last_successful_connection"].isoformat() if connection_state["last_successful_connection"] else None
                }
            })
            
        except Exception as e:
            logger.error(f"Error in connection monitor: {e}")
            connection_state["is_stable"] = False
            bot_state["connection_stable"] = False
            bot_state["connection_status"] = f"Error: {str(e)}"
        
        # Check every 30 seconds instead of 10
        await asyncio.sleep(30)

async def ensure_stable_connection():
    """Ensure we have a stable connection before operations"""
    max_wait = 30  # Maximum wait time in seconds
    wait_time = 0
    
    while not deriv_api.is_connected and wait_time < max_wait:
        logger.info("Waiting for stable connection...")
        try:
            await deriv_api.connect()
            if deriv_api.is_connected:
                logger.info("✅ Stable connection established")
                return True
        except Exception as e:
            logger.warning(f"Connection attempt failed: {e}")
        
        await asyncio.sleep(2)
        wait_time += 2
    
    if not deriv_api.is_connected:
        logger.error("❌ Could not establish stable connection")
        return False
    
    return True

if __name__ == "__main__":
    print("🌐 Starting Enhanced Deriv Trading Dashboard...")
    print("📊 Dashboard URL: http://localhost:8000")
    print("🔄 Real-time updates enabled")
    print("🎮 Trading controls available")
    print("🤖 Professional AI Trading Dashboard")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
