import React, { useState, useEffect } from 'react';
import { Button, Card, Form, Alert, Row, Col, Badge } from 'react-bootstrap';

const RiskManagementPanel = ({ botStatus, onResetSessionCounter, onDisableSessionLimit }) => {
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');

  const handleResetSessionCounter = async () => {
    try {
      await onResetSessionCounter();
      setAlertMessage('Session trade counter reset successfully');
      setShowSuccessAlert(true);
      setTimeout(() => setShowSuccessAlert(false), 3000);
    } catch (error) {
      console.error('Error resetting session counter:', error);
    }
  };

  const handleDisableSessionLimit = async () => {
    try {
      await onDisableSessionLimit();
      setAlertMessage('Session trade limit disabled');
      setShowSuccessAlert(true);
      setTimeout(() => setShowSuccessAlert(false), 3000);
    } catch (error) {
      console.error('Error disabling session limit:', error);
    }
  };

  // Risk management information from bot status
  const riskInfo = botStatus?.risk_management || {
    max_trades_per_session: 100,  // Updated default to 100
    current_trades: 0,
    max_consecutive_losses: 50,   // Updated from 10 to 50
    current_consecutive_losses: 0,
    limits_hit: false
  };
  
  // Check if session limit is disabled
  const isSessionLimitDisabled = riskInfo.max_trades_per_session === 0;

  return (
    <Card className="mb-4">
      <Card.Header className="d-flex justify-content-between align-items-center">
        <h5 className="mb-0">Risk Management</h5>
        {riskInfo.limits_hit && (
          <Badge bg="warning" className="ms-2">Limit Hit</Badge>
        )}
      </Card.Header>
      <Card.Body>
        {showSuccessAlert && (
          <Alert variant="success" onClose={() => setShowSuccessAlert(false)} dismissible>
            {alertMessage}
          </Alert>
        )}
        
        <Row className="mb-3">
          <Col>
            <div className="d-flex justify-content-between">
              <span>Session Trades:</span>
              <span className={riskInfo.current_trades >= riskInfo.max_trades_per_session && !isSessionLimitDisabled ? 'text-danger' : ''}>
                {riskInfo.current_trades} {isSessionLimitDisabled ? '/ ∞' : `/ ${riskInfo.max_trades_per_session}`}
              </span>
            </div>
          </Col>
          <Col>
            <div className="d-flex justify-content-between">
              <span>Consecutive Losses:</span>
              <span className={riskInfo.current_consecutive_losses >= riskInfo.max_consecutive_losses ? 'text-danger' : ''}>
                {riskInfo.current_consecutive_losses} / {riskInfo.max_consecutive_losses}
              </span>
            </div>
          </Col>
        </Row>
        
        {riskInfo.limits_hit && (
          <Alert variant="warning">
            Risk limit hit: {riskInfo.limits_hit_reason || 'Unknown reason'}
          </Alert>
        )}
        
        <div className="mt-3 d-flex gap-2">
          <Button 
            variant="warning" 
            size="sm" 
            onClick={handleResetSessionCounter}
            disabled={riskInfo.current_trades === 0}
          >
            Reset Trade Counter
          </Button>
          <Button 
            variant={isSessionLimitDisabled ? "outline-success" : "outline-danger"}
            size="sm" 
            onClick={handleDisableSessionLimit}
          >
            {isSessionLimitDisabled ? "Enable Trade Limit" : "Disable Trade Limit"}
          </Button>
        </div>
      </Card.Body>
    </Card>
  );
};

export default RiskManagementPanel;
