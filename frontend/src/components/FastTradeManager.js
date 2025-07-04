import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    Box, Card, CardContent, Typography, Grid, Switch, 
    Button, Alert, CircularProgress, Divider,
    Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';
import { BACKEND_API_URL } from '../config';
import ActiveTradesTable from './ActiveTradesTable';

const FastTradeManager = () => {
    const { token } = useAuth();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [botStatus, setBotStatus] = useState({});
    const [activeTrades, setActiveTrades] = useState([]);
    const [tradeHistory, setTradeHistory] = useState([]);
    const [autoRefresh, setAutoRefresh] = useState(true);

    // Fetch bot status, active trades, and trade history
    const fetchBotData = async () => {
        try {
            // Get bot status
            const statusResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/status`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (statusResponse.data.success) {
                setBotStatus(statusResponse.data.status);
            }

            // Get active trades
            const tradesResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/active-trades`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (tradesResponse.data.success) {
                setActiveTrades(tradesResponse.data.trades);
            }

            // Get recent trade history
            const historyResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/trade-history`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (historyResponse.data.success) {
                setTradeHistory(historyResponse.data.trades);
            }

            setLoading(false);
        } catch (err) {
            console.error("Error fetching bot data:", err);
            setError("Failed to load bot data. Please try again later.");
            setLoading(false);
        }
    };

    // Start bot
    const startBot = async () => {
        try {
            await axios.post(
                `${BACKEND_API_URL}/trading-bot/start`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            fetchBotData();
        } catch (err) {
            console.error("Error starting bot:", err);
            setError("Failed to start bot. Please try again later.");
        }
    };

    // Stop bot
    const stopBot = async () => {
        try {
            await axios.post(
                `${BACKEND_API_URL}/trading-bot/stop`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            fetchBotData();
        } catch (err) {
            console.error("Error stopping bot:", err);
            setError("Failed to stop bot. Please try again later.");
        }
    };

    // Force close a trade
    const handleForceClose = async (tradeId) => {
        try {
            await axios.post(
                `${BACKEND_API_URL}/trading-bot/force-close/${tradeId}`,
                {},
                { headers: { Authorization: `Bearer ${token}` } }
            );
            fetchBotData();
        } catch (err) {
            console.error("Error closing trade:", err);
            setError("Failed to close trade. Please try again later.");
        }
    };

    // Auto refresh
    useEffect(() => {
        fetchBotData();

        const interval = autoRefresh ? setInterval(fetchBotData, 3000) : null;
        
        return () => {
            if (interval) {
                clearInterval(interval);
            }
        };
    }, [autoRefresh, token]);

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
                <CircularProgress />
            </Box>
        );
    }

    return (
        <Box sx={{ p: 2 }}>
            {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                </Alert>
            )}

            {/* Bot Control Panel */}
            <Card sx={{ mb: 4 }}>
                <CardContent>
                    <Grid container spacing={2} alignItems="center">
                        <Grid item xs={12} md={6}>
                            <Typography variant="h5">Fast Trading Bot Control</Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                Executes trades every 5-10 seconds automatically
                            </Typography>
                        </Grid>
                        
                        <Grid item xs={6} md={3}>
                            <Button 
                                variant="contained" 
                                color="primary" 
                                fullWidth
                                disabled={botStatus.is_running}
                                onClick={startBot}
                            >
                                Start Bot
                            </Button>
                        </Grid>
                        
                        <Grid item xs={6} md={3}>
                            <Button 
                                variant="contained" 
                                color="error" 
                                fullWidth
                                disabled={!botStatus.is_running}
                                onClick={stopBot}
                            >
                                Stop Bot
                            </Button>
                        </Grid>
                    </Grid>

                    <Box sx={{ mt: 3 }}>
                        <Typography variant="subtitle2" gutterBottom>
                            Bot Status: 
                            <span style={{ 
                                color: botStatus.is_running ? 'green' : 'red',
                                marginLeft: '8px'
                            }}>
                                {botStatus.is_running ? 'Running' : 'Stopped'}
                            </span>
                        </Typography>

                        <Grid container spacing={2} sx={{ mt: 1 }}>
                            <Grid item xs={6} sm={4} md={2}>
                                <Typography variant="body2" color="text.secondary">Current Balance</Typography>
                                <Typography variant="h6">${botStatus.account_balance?.toFixed(2) || '0.00'}</Typography>
                            </Grid>
                            <Grid item xs={6} sm={4} md={2}>
                                <Typography variant="body2" color="text.secondary">Daily Profit</Typography>
                                <Typography variant="h6" color="success.main">
                                    ${botStatus.daily_profit?.toFixed(2) || '0.00'}
                                </Typography>
                            </Grid>
                            <Grid item xs={6} sm={4} md={2}>
                                <Typography variant="body2" color="text.secondary">Daily Loss</Typography>
                                <Typography variant="h6" color="error.main">
                                    ${botStatus.daily_loss?.toFixed(2) || '0.00'}
                                </Typography>
                            </Grid>
                            <Grid item xs={6} sm={4} md={2}>
                                <Typography variant="body2" color="text.secondary">Win Rate</Typography>
                                <Typography variant="h6">{botStatus.win_rate?.toFixed(1) || '0.0'}%</Typography>
                            </Grid>
                            <Grid item xs={6} sm={4} md={2}>
                                <Typography variant="body2" color="text.secondary">Trades Today</Typography>
                                <Typography variant="h6">{botStatus.total_trades_today || 0}</Typography>
                            </Grid>
                            <Grid item xs={6} sm={4} md={2}>
                                <Typography variant="body2" color="text.secondary">Auto Refresh</Typography>
                                <Switch 
                                    checked={autoRefresh} 
                                    onChange={(e) => setAutoRefresh(e.target.checked)} 
                                    color="primary" 
                                />
                            </Grid>
                        </Grid>
                    </Box>
                </CardContent>
            </Card>

            {/* Active Trades */}
            <Typography variant="h6" gutterBottom>Active Trades</Typography>
            <ActiveTradesTable trades={activeTrades} onForceClose={handleForceClose} />

            {/* Recent Trade History */}
            <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>Recent Trade History</Typography>
            <TableContainer component={Paper}>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Direction</TableCell>
                            <TableCell>Entry Price</TableCell>
                            <TableCell>Exit Price</TableCell>
                            <TableCell>Stake</TableCell>
                            <TableCell>P/L</TableCell>
                            <TableCell>Status</TableCell>
                            <TableCell>Time</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {tradeHistory.slice(0, 10).map((trade) => (
                            <TableRow key={trade.id} hover>
                                <TableCell>
                                    <span style={{ 
                                        color: trade.direction === 'RISE' ? 'green' : 'red',
                                        fontWeight: 'bold'
                                    }}>
                                        {trade.direction}
                                    </span>
                                </TableCell>
                                <TableCell>${trade.entry_price.toFixed(5)}</TableCell>
                                <TableCell>${trade.exit_price?.toFixed(5) || '-'}</TableCell>
                                <TableCell>${trade.stake.toFixed(2)}</TableCell>
                                <TableCell style={{ 
                                    color: trade.profit_loss > 0 ? 'green' : 
                                           trade.profit_loss < 0 ? 'red' : 'inherit'
                                }}>
                                    {trade.profit_loss > 0 ? '+' : ''}{trade.profit_loss.toFixed(2)}
                                </TableCell>
                                <TableCell>
                                    <span style={{
                                        color: trade.status === 'WON' ? 'green' : 
                                               trade.status === 'LOST' ? 'red' : 'inherit'
                                    }}>
                                        {trade.status}
                                    </span>
                                </TableCell>
                                <TableCell>
                                    {new Date(trade.entry_time).toLocaleTimeString()}
                                </TableCell>
                            </TableRow>
                        ))}
                        {tradeHistory.length === 0 && (
                            <TableRow>
                                <TableCell colSpan={7} align="center">No trade history available</TableCell>
                            </TableRow>
                        )}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
};

export default FastTradeManager;
