"""
Web Dashboard for monitoring the trading bot
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json
import asyncio
import os
from datetime import datetime, timedelta
from database import SessionLocal, Trade, TradingSession, AIAnalysis
from sqlalchemy import desc
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

app = FastAPI(title="Deriv AI Trading Bot Dashboard")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    """Get trading statistics"""
    try:
        db = SessionLocal()
        try:
            # Get recent trades
            recent_trades = db.query(Trade).order_by(desc(Trade.created_at)).limit(100).all()
            
            # Calculate statistics
            total_trades = len(recent_trades)
            winning_trades = len([t for t in recent_trades if t.profit_loss and t.profit_loss > 0])
            total_profit = sum(t.profit_loss or 0 for t in recent_trades)
            
            # Calculate averages
            avg_stake = sum(t.stake or 0 for t in recent_trades) / max(1, total_trades)
            avg_profit = total_profit / max(1, total_trades)
            avg_ai_confidence = sum(t.ai_confidence or 0 for t in recent_trades) / max(1, total_trades)
            
            # Get current session
            current_session = db.query(TradingSession).filter(
                TradingSession.is_active == True
            ).first()
            
            # Daily performance
            today = datetime.now().date()
            daily_trades = [t for t in recent_trades if t.created_at.date() == today]
            daily_profit = sum(t.profit_loss or 0 for t in daily_trades)
            
            # Mock live trades (in real implementation, get from active trades)
            live_trades = []
            active_trades = [t for t in recent_trades if t.status == 'PENDING']
            for trade in active_trades[:5]:  # Show up to 5 active trades
                live_trades.append({
                    'symbol': trade.symbol,
                    'trade_type': trade.trade_type,
                    'stake': trade.stake,
                    'ai_confidence': trade.ai_confidence or 0.5,
                    'duration': trade.duration or 5,
                    'entry_price': trade.entry_price,
                    'start_time': trade.start_time.isoformat() if trade.start_time else datetime.now().isoformat()
                })
            
            # Get current martingale level (mock implementation)
            current_martingale_level = 0
            if recent_trades:
                last_trade = recent_trades[0]
                current_martingale_level = last_trade.martingale_level or 0
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "win_rate": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                "total_profit": total_profit,
                "daily_trades": len(daily_trades),
                "daily_profit": daily_profit,
                "avg_stake": avg_stake,
                "avg_profit": avg_profit,
                "avg_ai_confidence": avg_ai_confidence,
                "current_martingale_level": current_martingale_level,
                "account_balance": 10000.0,  # Mock balance - in real implementation, get from Deriv API
                "live_trades": live_trades,
                "current_session": {
                    "id": current_session.session_id if current_session else None,
                    "start_time": current_session.start_time.isoformat() if current_session else None,
                    "total_trades": current_session.total_trades if current_session else 0,
                    "total_profit": current_session.total_profit_loss if current_session else 0
                }
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    try:
        db = SessionLocal()
        try:
            trades = db.query(Trade).order_by(desc(Trade.created_at)).limit(limit).all()
            
            return [{
                "id": trade.id,
                "contract_id": trade.contract_id,
                "symbol": trade.symbol,
                "trade_type": trade.trade_type,
                "stake": trade.stake,
                "profit_loss": trade.profit_loss,
                "status": trade.status,
                "ai_confidence": trade.ai_confidence,
                "martingale_level": trade.martingale_level,
                "created_at": trade.created_at.isoformat(),
                "end_time": trade.end_time.isoformat() if trade.end_time else None
            } for trade in trades]
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {"error": str(e)}

@app.get("/api/analysis")
async def get_analysis(limit: int = 20):
    """Get recent AI analysis"""
    try:
        db = SessionLocal()
        try:
            analyses = db.query(AIAnalysis).order_by(desc(AIAnalysis.timestamp)).limit(limit).all()
            
            return [{
                "id": analysis.id,
                "symbol": analysis.symbol,
                "timeframe": analysis.timeframe,
                "prediction": analysis.prediction,
                "confidence": analysis.confidence,
                "timestamp": analysis.timestamp.isoformat(),
                "analysis_text": json.loads(analysis.analysis_text) if analysis.analysis_text else {}
            } for analysis in analyses]
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting analysis: {e}")
        return {"error": str(e)}

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics"""
    try:
        db = SessionLocal()
        try:
            # Get trades from last 30 days
            thirty_days_ago = datetime.now() - timedelta(days=30)
            trades = db.query(Trade).filter(
                Trade.created_at >= thirty_days_ago,
                Trade.status.in_(['WON', 'LOST'])
            ).order_by(Trade.created_at).all()
            
            # Group by day
            daily_performance = {}
            for trade in trades:
                date_key = trade.created_at.date().isoformat()
                if date_key not in daily_performance:
                    daily_performance[date_key] = {
                        'trades': 0,
                        'profit': 0,
                        'wins': 0
                    }
                
                daily_performance[date_key]['trades'] += 1
                daily_performance[date_key]['profit'] += trade.profit_loss or 0
                if trade.profit_loss and trade.profit_loss > 0:
                    daily_performance[date_key]['wins'] += 1
            
            # Convert to chart data
            chart_data = []
            for date, data in sorted(daily_performance.items()):
                chart_data.append({
                    'date': date,
                    'trades': data['trades'],
                    'profit': data['profit'],
                    'win_rate': (data['wins'] / data['trades'] * 100) if data['trades'] > 0 else 0
                })
            
            return {
                'daily_performance': chart_data,
                'total_days': len(chart_data),
                'total_trades': sum(d['trades'] for d in chart_data),
                'total_profit': sum(d['profit'] for d in chart_data)
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(5)
            
            # Get latest stats and send to client
            stats = await get_stats()
            await websocket.send_text(json.dumps({
                "type": "stats_update",
                "data": stats
            }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Broadcast function for external use
async def broadcast_update(update_type: str, data: dict):
    """Broadcast update to all connected clients"""
    message = json.dumps({
        "type": update_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    })
    await manager.broadcast(message)

bot_process = None

@app.post("/api/start-bot")
async def start_bot(background_tasks: BackgroundTasks):
    """Start the trading bot as a background process"""
    global bot_process
    
    # Check if bot is already running
    if bot_process is not None and bot_process.poll() is None:
        return {"status": "already_running", "message": "Bot is already running"}
    
    try:
        # Check if main.py exists
        if not os.path.exists("main.py"):
            return {"status": "error", "detail": "main.py not found"}
        
        # Start the bot in a subprocess with proper working directory
        bot_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        await asyncio.sleep(1)
        
        # Check if it's still running (didn't crash immediately)
        if bot_process.poll() is None:
            logger.info("✅ Trading bot started successfully")
            return {"status": "started", "message": "Trading bot started successfully", "pid": bot_process.pid}
        else:
            # Bot crashed immediately, get error output
            stdout, stderr = bot_process.communicate()
            error_msg = stderr if stderr else stdout if stdout else "Unknown error"
            logger.error(f"❌ Bot failed to start: {error_msg}")
            return {"status": "error", "detail": f"Bot failed to start: {error_msg}"}
            
    except Exception as e:
        logger.error(f"❌ Error starting bot: {e}")
        return {"status": "error", "detail": f"Failed to start bot: {str(e)}"}

@app.get("/api/bot-status")
async def get_bot_status():
    """Get current bot status"""
    global bot_process
    
    if bot_process is None:
        return {"status": "not_running", "message": "Bot is not running"}
    
    # Check if process is still alive
    if bot_process.poll() is None:
        return {"status": "running", "message": "Bot is running", "pid": bot_process.pid}
    else:
        # Process has ended, clean up
        bot_process = None
        return {"status": "stopped", "message": "Bot has stopped"}

@app.post("/api/stop-bot")
async def stop_bot():
    """Stop the trading bot process"""
    global bot_process
    if bot_process is not None and bot_process.poll() is None:
        bot_process.terminate()
        bot_process = None
        return {"status": "stopped"}
    return {"status": "not_running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
