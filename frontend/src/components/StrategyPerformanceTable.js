import React from 'react';
import { 
    Table, TableBody, TableCell, TableContainer, TableHead, 
    TableRow, Paper, Chip, Typography, Box
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';

const StrategyPerformanceTable = ({ strategyStats, strategies }) => {
    // Helper function to format profit/loss display
    const formatProfit = (profit) => {
        if (profit > 0) {
            return <span style={{ color: 'green' }}>+${profit.toFixed(2)}</span>;
        } else if (profit < 0) {
            return <span style={{ color: 'red' }}>-${Math.abs(profit).toFixed(2)}</span>;
        }
        return <span>$0.00</span>;
    };

    // Helper function to get strategy name from ID
    const getStrategyName = (id) => {
        const strategy = strategies.find(s => s.id === id);
        return strategy ? strategy.name : `Strategy ${id}`;
    };

    // Get sorted stats array for display
    const sortedStats = Object.entries(strategyStats)
        .map(([id, stats]) => ({
            id: parseInt(id),
            ...stats,
            name: getStrategyName(parseInt(id))
        }))
        .sort((a, b) => {
            // Sort by active status first, then by win rate
            if (a.status === 'Active' && b.status !== 'Active') return -1;
            if (a.status !== 'Active' && b.status === 'Active') return 1;
            return b.win_rate - a.win_rate;
        });

    return (
        <Box sx={{ mt: 2 }}>
            <Typography variant="h6" gutterBottom>Strategy Performance</Typography>
            <TableContainer component={Paper}>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                            <TableCell>Strategy</TableCell>
                            <TableCell align="center">Status</TableCell>
                            <TableCell align="center">Trades</TableCell>
                            <TableCell align="center">Win Rate</TableCell>
                            <TableCell align="center">Wins</TableCell>
                            <TableCell align="center">Losses</TableCell>
                            <TableCell align="center">Profit/Loss</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {sortedStats.map((stat) => (
                            <TableRow key={stat.id}>
                                <TableCell component="th" scope="row">
                                    <Typography variant="body2" noWrap>
                                        {stat.name}
                                    </Typography>
                                </TableCell>
                                <TableCell align="center">
                                    <Chip 
                                        size="small"
                                        label={stat.status} 
                                        color={stat.status === 'Active' ? "success" : 
                                               stat.status === 'Signal Detected' ? "warning" : "default"}
                                    />
                                </TableCell>
                                <TableCell align="center">{stat.trades}</TableCell>
                                <TableCell align="center">
                                    {stat.trades > 0 ? `${stat.win_rate}%` : 'N/A'}
                                </TableCell>
                                <TableCell align="center">
                                    <Box display="flex" alignItems="center" justifyContent="center">
                                        <CheckCircleIcon fontSize="small" sx={{ color: 'green', mr: 0.5 }} />
                                        {stat.wins}
                                    </Box>
                                </TableCell>
                                <TableCell align="center">
                                    <Box display="flex" alignItems="center" justifyContent="center">
                                        <CancelIcon fontSize="small" sx={{ color: 'red', mr: 0.5 }} />
                                        {stat.losses}
                                    </Box>
                                </TableCell>
                                <TableCell align="center">
                                    {formatProfit(stat.profit)}
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
};

export default StrategyPerformanceTable;
