# DigitalOcean App Platform Configuration Guide

This guide provides specific instructions for deploying the Deriv Trading Bot on DigitalOcean App Platform.

## WebSocket Configuration (Critical)

WebSockets are essential for real-time trading updates. Follow these steps to ensure they work correctly:

### 1. App Specification

When setting up your App Platform application:

1. Create a Web Service for your backend:
   - Source: GitHub repository
   - Branch: main
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Run Command: `cd backend && python production_server.py`

2. Create a Static Site for your frontend:
   - Source: Same GitHub repository
   - Branch: main
   - Build Command: `cd frontend && npm install && npm run build`
   - Output Directory: `frontend/build`

### 2. HTTP Routes Configuration

Set up these HTTP routes in your App Platform dashboard:

1. **API Route**:
   - Route Path: `/api`
   - Component: Your backend service
   - Forward Path: `/api`
   - ✓ Preserve Path Prefix

2. **Socket.IO Route** (critical):
   - Route Path: `/socket.io`
   - Component: Your backend service
   - Forward Path: `/socket.io`
   - ✓ Preserve Path Prefix

3. **API Socket.IO Route**:
   - Route Path: `/api/socket.io`
   - Component: Your backend service
   - Forward Path: `/socket.io`
   - ✓ Preserve Path Prefix

![DigitalOcean App Platform Routes Configuration Example](https://i.imgur.com/example.png)

### 3. Environment Variables

Configure these environment variables for your components:

#### Backend Service:
