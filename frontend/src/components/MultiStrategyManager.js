import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
    Box, Card, CardContent, Typography, Grid, Switch, 
    Table, TableBody, TableCell, TableContainer, TableHead, 
    TableRow, Paper, Chip, CircularProgress, Button,
    Accordion, AccordionSummary, AccordionDetails, Alert,
    Divider
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import { useAuth } from '../contexts/AuthContext';
import { BACKEND_API_URL } from '../config';

const MultiStrategyManager = () => {
    const { token } = useAuth();
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [strategies, setStrategies] = useState([]);
    const [activeStrategies, setActiveStrategies] = useState({});
    const [strategyStats, setStrategyStats] = useState({});
    const [strategyTrades, setStrategyTrades] = useState({});
    const [activeTrades, setActiveTrades] = useState({});
    const [botRunning, setBotRunning] = useState(false);
    const [autoRefresh, setAutoRefresh] = useState(true);

    // Fetch all strategies and their status
    const fetchStrategies = async () => {
        try {
            // Get available strategies
            const strategiesResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/strategies`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (strategiesResponse.data.success) {
                setStrategies(strategiesResponse.data.strategies);
            }

            // Get strategy status
            const statusResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/strategies-status`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (statusResponse.data.success) {
                setActiveStrategies(statusResponse.data.data.active_strategies);
                setStrategyStats(statusResponse.data.data.strategy_stats || {});
                
                // Get strategy active trades if available in the response
                if (statusResponse.data.data.strategy_active_trades) {
                    setStrategyTrades(statusResponse.data.data.strategy_active_trades);
                }
            }

            // Get bot status
            const botStatusResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/status`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (botStatusResponse.data.success) {
                setBotRunning(botStatusResponse.data.status.is_running);
            }

            // Get all strategy trades
            const tradesResponse = await axios.get(
                `${BACKEND_API_URL}/trading-bot/all-strategy-trades`,
                { headers: { Authorization: `Bearer ${token}` } }
            );

            if (tradesResponse.data.success) {
                setStrategyTrades(tradesResponse.data.strategy_trades);
            }

            setLoading(false);
        } catch (err) {
            console.error("Error fetching strategy data:", err);
            setError("Failed to load strategies. Please try again later.");
            setLoading(false);
        }
    };

    // Toggle strategy active status
    const toggleStrategy = async (strategyId, active) => {
        try {
            await axios.post(
                `${BACKEND_API_URL}/trading-bot/strategy/${strategyId}/toggle`,
                { active },
                { headers: { Authorization: `Bearer ${token}` } }
            );

            // Update local state
            setActiveStrategies(prev => ({
                ...prev,
                [strategyId]: active
            }));
        } catch (err) {
            console.error(`Error toggling strategy ${strategyId}:`, err);
            setError(`Failed to update strategy ${strategyId}. Please try again.`);
        }
    };

    // Auto refresh data
    useEffect(() => {
        fetchStrategies();

        let interval;
        if (autoRefresh) {
            interval = setInterval(() => {
                fetchStrategies();
            }, 3000); // Refresh every 3 seconds
        }

        return () => {
            if (interval) clearInterval(interval);
        };
    }, [token, autoRefresh]);

    // Format date
    const formatDate = (dateString) => {
        if (!dateString) return "N/A";
        const date = new Date(dateString);
        return date.toLocaleTimeString();
    };

    // Get strategy name from ID
    // Helper function to get strategy name from ID (used in rendering)
    const getStrategyNameById = (id) => {
        const strategy = strategies.find(s => s.id === id);
        return strategy ? strategy.name : `Strategy ${id}`;
    };

    // Get strategy chip color based on risk level
    const getRiskColor = (riskLevel) => {
        switch (riskLevel) {
            case 'low': return 'success';
            case 'medium': return 'warning';
            case 'medium-high': case 'high': return 'error';
            default: return 'default';
        }
    };

    // Get status color
    const getStatusColor = (status) => {
        switch (status) {
            case 'Active': return 'success';
            case 'Inactive': return 'default';
            case 'Signal Detected': return 'warning';
            case 'Analyzing': return 'info';
            default: return 'default';
        }
    };

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ p: 2 }}>
                <Alert severity="error">{error}</Alert>
            </Box>
        );
    }

    return (
        <Box sx={{ p: 2 }}>
            <Typography variant="h4" gutterBottom>
                Multi-Strategy Trading Manager
            </Typography>

            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">
                    Bot Status: <Chip 
                        color={botRunning ? "success" : "default"} 
                        label={botRunning ? "Running" : "Stopped"} 
                        size="small" 
                    />
                </Typography>
                <Box>
                    <Button 
                        variant="outlined" 
                        color="primary" 
                        onClick={fetchStrategies}
                        sx={{ mr: 2 }}
                    >
                        Refresh Data
                    </Button>
                    <Typography component="div" sx={{ display: 'inline-flex', alignItems: 'center' }}>
                        <span>Auto Refresh:</span>
                        <Switch
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                            color="primary"
                        />
                    </Typography>
                </Box>
            </Box>

            <Grid container spacing={2}>
                {/* Strategy Summary Cards */}
                <Grid item xs={12}>
                    <Typography variant="h5" gutterBottom>
                        Strategy Performance Summary
                    </Typography>
                    <Grid container spacing={2}>
                        {strategies.map((strategy) => {
                            const stats = strategyStats[strategy.id] || {};
                            const isActive = activeStrategies[strategy.id] || false;

                            return (
                                <Grid item xs={12} sm={6} md={4} lg={3} key={strategy.id}>
                                    <Card 
                                        sx={{ 
                                            height: '100%', 
                                            opacity: isActive ? 1 : 0.7,
                                            border: isActive ? '1px solid #4caf50' : '1px solid #ccc'
                                        }}
                                    >
                                        <CardContent>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                                <Typography variant="h6" component="div" noWrap sx={{ maxWidth: '70%' }}>
                                                    {strategy.name}
                                                </Typography>
                                                <Switch
                                                    checked={isActive}
                                                    onChange={(e) => toggleStrategy(strategy.id, e.target.checked)}
                                                    disabled={!botRunning}
                                                />
                                            </Box>
                                            
                                            <Box sx={{ mb: 1 }}>
                                                <Chip 
                                                    size="small" 
                                                    color={getRiskColor(strategy.risk_level)}
                                                    label={`Risk: ${strategy.risk_level}`} 
                                                    sx={{ mr: 1, mb: 1 }}
                                                />
                                                <Chip 
                                                    size="small" 
                                                    color={getStatusColor(stats.status)}
                                                    label={`Status: ${stats.status || 'Unknown'}`}
                                                    sx={{ mb: 1 }}
                                                />
                                            </Box>
                                            
                                            <TableContainer component={Paper} variant="outlined" sx={{ mb: 1 }}>
                                                <Table size="small">
                                                    <TableBody>
                                                        <TableRow>
                                                            <TableCell>Trades</TableCell>
                                                            <TableCell align="right">{stats.trades || 0}</TableCell>
                                                        </TableRow>
                                                        <TableRow>
                                                            <TableCell>Win Rate</TableCell>
                                                            <TableCell align="right">{stats.win_rate || 0}%</TableCell>
                                                        </TableRow>
                                                        <TableRow>
                                                            <TableCell>P&L</TableCell>
                                                            <TableCell 
                                                                align="right"
                                                                sx={{ 
                                                                    color: (stats.profit || 0) >= 0 ? 'green' : 'red',
                                                                    fontWeight: 'bold'
                                                                }}
                                                            >
                                                                ${Number(stats.profit || 0).toFixed(2)}
                                                            </TableCell>
                                                        </TableRow>
                                                    </TableBody>
                                                </Table>
                                            </TableContainer>
                                            
                                            <Typography variant="caption" color="text.secondary">
                                                {stats.last_signal_time ? 
                                                    `Last Signal: ${formatDate(stats.last_signal_time)}` : 
                                                    'No signals yet'}
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            );
                        })}
                    </Grid>
                </Grid>

                {/* Active Trades by Strategy */}
                <Grid item xs={12} sx={{ mt: 4 }}>
                    <Typography variant="h5" gutterBottom>
                        Current Active Trades by Strategy
                    </Typography>
                    
                    {Object.entries(strategyTrades).some(([_, trades]) => trades.length > 0) ? (
                        strategies.map((strategy) => {
                            const trades = strategyTrades[strategy.id] || [];
                            
                            if (trades.length === 0) return null;
                            
                            return (
                                <Accordion key={strategy.id} sx={{ mb: 2 }}>
                                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                                            <Typography sx={{ flexGrow: 1 }}>
                                                {strategy.name}
                                            </Typography>
                                            <Chip 
                                                label={`${trades.length} Active Trades`}
                                                color="primary" 
                                                size="small" 
                                                sx={{ mr: 2 }}
                                            />
                                        </Box>
                                    </AccordionSummary>
                                    <AccordionDetails>
                                        <TableContainer component={Paper} variant="outlined">
                                            <Table size="small">
                                                <TableHead>
                                                    <TableRow>
                                                        <TableCell>Trade ID</TableCell>
                                                        <TableCell>Direction</TableCell>
                                                        <TableCell>Stake</TableCell>
                                                        <TableCell>Entry Price</TableCell>
                                                        <TableCell align="center">Entry Time</TableCell>
                                                        <TableCell align="center">Duration</TableCell>
                                                        <TableCell align="center">Status</TableCell>
                                                    </TableRow>
                                                </TableHead>
                                                <TableBody>
                                                    {trades.map((trade) => (
                                                        <TableRow key={trade.id}>
                                                            <TableCell>{trade.id}</TableCell>
                                                            <TableCell>
                                                                {trade.direction === "RISE" ? (
                                                                    <Chip 
                                                                        icon={<TrendingUpIcon />} 
                                                                        label="Rise" 
                                                                        size="small"
                                                                        color="success"
                                                                    />
                                                                ) : (
                                                                    <Chip 
                                                                        icon={<TrendingDownIcon />} 
                                                                        label="Fall" 
                                                                        size="small"
                                                                        color="error"
                                                                    />
                                                                )}
                                                            </TableCell>
                                                            <TableCell>${trade.stake}</TableCell>
                                                            <TableCell>{trade.entry_price}</TableCell>
                                                            <TableCell align="center">{new Date(trade.entry_time).toLocaleTimeString()}</TableCell>
                                                            <TableCell align="center">{trade.duration}s</TableCell>
                                                            <TableCell align="center">
                                                                {trade.status === "ACTIVE" && <Chip 
                                                                    label="Active" 
                                                                    color="primary" 
                                                                    size="small"
                                                                />}
                                                                {trade.status === "WON" && <Chip 
                                                                    label="Won" 
                                                                    color="success" 
                                                                    size="small"
                                                                    icon={<CheckCircleIcon />}
                                                                />}
                                                                {trade.status === "LOST" && <Chip 
                                                                    label="Lost" 
                                                                    color="error" 
                                                                    size="small"
                                                                    icon={<CancelIcon />}
                                                                />}
                                                                {trade.status === "PENDING" && <Chip 
                                                                    label="Pending" 
                                                                    color="warning" 
                                                                    size="small"
                                                                    icon={<HourglassEmptyIcon />}
                                                                />}
                                                            </TableCell>
                                                        </TableRow>
                                                    ))}
                                                </TableBody>
                                            </Table>
                                        </TableContainer>
                                    </AccordionDetails>
                                </Accordion>
                            );
                        })
                    ) : (
                        <Typography variant="body1" color="text.secondary" sx={{ mt: 2 }}>
                            No active trades at the moment
                        </Typography>
                    )}
                </Grid>
            </Grid>

            {/* Multi-Strategy Dashboard */}
            <Box sx={{ mt: 4 }}>
                <Typography variant="h5" gutterBottom>Multi-Strategy Dashboard</Typography>
                <Alert severity="info" sx={{ mb: 2 }}>
                    This dashboard shows real-time performance of all active trading strategies.
                </Alert>

                <Grid container spacing={2}>
                    {/* Summary Stats Card */}
                    <Grid item xs={12} md={4}>
                        <Card variant="outlined">
                            <CardContent>
                                <Typography variant="h6" gutterBottom>Summary</Typography>
                                <Typography variant="body1">
                                    Active Strategies: {Object.values(activeStrategies).filter(Boolean).length} / {strategies.length}
                                </Typography>
                                <Typography variant="body1">
                                    Active Trades: {Object.values(activeTrades || {}).length}
                                </Typography>
                                <Typography variant="body1">
                                    Total Trades Today: {Object.values(strategyStats).reduce((sum, stat) => sum + stat.trades, 0)}
                                </Typography>
                                <Typography variant="body1">
                                    Overall Win Rate: {
                                        (() => {
                                            const totalWins = Object.values(strategyStats).reduce((sum, stat) => sum + stat.wins, 0);
                                            const totalLosses = Object.values(strategyStats).reduce((sum, stat) => sum + stat.losses, 0);
                                            const totalTrades = totalWins + totalLosses;
                                            return totalTrades > 0 ? ((totalWins / totalTrades) * 100).toFixed(1) + '%' : 'N/A';
                                        })()
                                    }
                                </Typography>
                            </CardContent>
                        </Card>
                    </Grid>

                    {/* Top Performing Strategies */}
                    <Grid item xs={12} md={8}>
                        <Card variant="outlined">
                            <CardContent>
                                <Typography variant="h6" gutterBottom>Top Performing Strategies</Typography>
                                <TableContainer>
                                    <Table size="small">
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>Strategy</TableCell>
                                                <TableCell align="center">Win Rate</TableCell>
                                                <TableCell align="center">Trades</TableCell>
                                                <TableCell align="right">Profit</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {Object.entries(strategyStats)
                                                .filter(([id, stat]) => stat.trades > 0)
                                                .sort(([_, a], [__, b]) => b.win_rate - a.win_rate)
                                                .slice(0, 5)
                                                .map(([id, stat]) => (
                                                    <TableRow key={id}>
                                                        <TableCell>
                                                            {getStrategyNameById(parseInt(id))}
                                                        </TableCell>
                                                        <TableCell align="center">
                                                            {stat.win_rate.toFixed(1)}%
                                                        </TableCell>
                                                        <TableCell align="center">{stat.trades}</TableCell>
                                                        <TableCell align="right" 
                                                            sx={{ color: stat.profit >= 0 ? 'success.main' : 'error.main' }}>
                                                            {stat.profit >= 0 ? '+' : ''}{stat.profit.toFixed(2)}
                                                        </TableCell>
                                                    </TableRow>
                                                ))
                                            }
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </CardContent>
                        </Card>
                    </Grid>
                </Grid>
            </Box>
    </Box>
);
};

export default MultiStrategyManager;
