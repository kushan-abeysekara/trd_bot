# Deployment Guide - Deriv Trading Bot

This guide will help you fix the "Invalid Host header" error and deploy the trading bot on your server.

## Quick Fix for "Invalid Host header"

The error occurs because the React development server blocks requests from external hosts for security. Here are the solutions:

### Option 1: Use the Production Scripts (Recommended)

1. **For Windows:**
   ```cmd
   start-production.bat
   ```

2. **For Linux/Mac:**
   ```bash
   chmod +x start-production.sh
   ./start-production.sh
   ```

### Option 2: Manual Setup

1. **Backend Setup:**
   ```bash
   cd backend
   pip install -r requirements.txt
   python production_server.py
   ```

2. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   npm run build
   npm run start-production
   ```

## Environment Configuration

The application now uses environment variables for configuration:

### Backend Environment (.env files in backend/)
- `.env.development` - For local development
- `.env.production` - For production deployment

### Frontend Environment (.env files in frontend/)
- `.env` - Default configuration
- `.env.development` - Development settings
- `.env.production` - Production settings

## Production Deployment

### 1. VPS/Server Deployment

1. **Upload files to your server**
2. **Install dependencies:**
   ```bash
   # Backend
   cd backend
   pip install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   npm run build
   ```

3. **Configure environment:**
   - Update `backend/.env.production` with your domain
   - Update `frontend/.env.production` with your domain

4. **Start services:**
   ```bash
   # Backend (production mode)
   cd backend
   FLASK_ENV=production python production_server.py &
   
   # Frontend (serve built files)
   cd ../frontend
   npm run start-production &
   ```

### 2. Using PM2 (Process Manager)

```bash
# Install PM2
npm install -g pm2

# Start backend
cd backend
pm2 start production_server.py --name "deriv-bot-backend" --interpreter python3

# Start frontend
cd ../frontend
pm2 start "npm run start-production" --name "deriv-bot-frontend"

# Save PM2 configuration
pm2 save
pm2 startup
```

### 3. Using Docker (Optional)

```dockerfile
# Create Dockerfile in project root
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/ ./backend/
COPY --from=frontend-build /app/frontend/build ./frontend/build

EXPOSE 5000 8080
CMD ["python", "backend/production_server.py"]
```

## Firewall Configuration

Make sure these ports are open:
- **5000** - Backend API
- **8080** - Frontend
- **80** - HTTP (if using Nginx)
- **443** - HTTPS (if using SSL)

### Ubuntu/Debian:
```bash
sudo ufw allow 5000
sudo ufw allow 8080
sudo ufw allow 80
sudo ufw allow 443
```

### CentOS/RHEL:
```bash
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --reload
```

## Nginx Configuration (Optional)

For production, you can use Nginx as a reverse proxy:

1. **Install Nginx:**
   ```bash
   sudo apt update
   sudo apt install nginx
   ```

2. **Copy configuration:**
   ```bash
   sudo cp nginx.conf /etc/nginx/sites-available/deriv-trading-bot
   sudo ln -s /etc/nginx/sites-available/deriv-trading-bot /etc/nginx/sites-enabled/
   ```

3. **Update domain name in nginx.conf**

4. **Test and restart:**
   ```bash
   sudo nginx -t
   sudo systemctl restart nginx
   ```

## Troubleshooting

### "Invalid Host header" Error
- Make sure you're using the production scripts
- Check that `DANGEROUSLY_DISABLE_HOST_CHECK=true` is set in frontend/.env
- Verify HOST=0.0.0.0 in frontend environment files

### Connection Issues
- Check firewall settings
- Verify correct ports are open
- Ensure services are running on correct hosts (0.0.0.0, not localhost)

### CORS Errors
- Update CORS_ORIGINS in backend/.env.production
- Include your domain in allowed origins
- Check protocol (http vs https) matches

## Production URLs

After deployment, your application will be available at:
- **Frontend:** `http://your-server-ip:8080`
- **Backend API:** `http://your-server-ip:5000/api`

## Security Considerations

1. **Use HTTPS in production**
2. **Set strong Flask secret key**
3. **Restrict CORS origins to your domain only**
4. **Use environment variables for sensitive data**
5. **Keep API tokens secure**
6. **Regular security updates**

For questions or support, check the main README.md file.
    proxy_send_timeout 3600s;
}
```

### 2. DigitalOcean App Platform Configuration

If using DigitalOcean App Platform:

1. Add a new HTTP Route:
   - Route Path: `/socket.io`
   - Rewrite Path: (leave blank)
   - ✓ Tick "Preserve Path Prefix"

2. Also add another route if using API prefix:
   - Route Path: `/api/socket.io`
   - Rewrite Path: `/socket.io`
   - ✓ Tick "Preserve Path Prefix"

3. Ensure WebSocket support is enabled in your App Platform settings

### 3. Frontend Socket.IO Configuration

Update your frontend Socket.IO connection to use the correct path:

```javascript
// In your React app
const socketOptions = {
  path: API_BASE_URL.includes('/api') ? '/api/socket.io' : '/socket.io',
  transports: ['websocket', 'polling'],
  reconnectionAttempts: 5
};

const socket = io(WS_URL, socketOptions);
```

## Troubleshooting

### "Invalid Host header" Error
- Make sure you're using the production scripts
- Check that `DANGEROUSLY_DISABLE_HOST_CHECK=true` is set in frontend/.env
- Verify HOST=0.0.0.0 in frontend environment files

### Connection Issues
- Check firewall settings
- Verify correct ports are open
- Ensure services are running on correct hosts (0.0.0.0, not localhost)

### CORS Errors
- Update CORS_ORIGINS in backend/.env.production
- Include your domain in allowed origins
- Check protocol (http vs https) matches

### WebSocket Issues
- Ensure Socket.IO is properly configured
- Check that WebSocket connections aren't blocked by proxy/firewall
- Verify correct WebSocket URL in frontend configuration
- Check WebSocket network traffic in browser DevTools (WS tab)
- Test WebSocket connectivity using the /api/socket-test endpoint

#### WebSocket Debug Steps:
1. Check browser console for connection errors
2. In DevTools Network tab, filter by WS to see if connection attempts are made
3. Verify Upgrade headers are present in the request/response
4. Test with both WebSocket and polling transports

## WebSocket Troubleshooting

If you're experiencing WebSocket connection issues ("Invalid session", 400 errors, or failure to establish WebSocket connections), try these solutions:

### 1. Update Frontend Socket.IO Configuration

Change your Socket.IO initialization to use WebSocket transport only (no polling):

```javascript
// In your React app's useEffect
const socketOptions = {
  path: API_BASE_URL.includes('/api') ? '/api/socket.io' : '/socket.io',
  transports: ['websocket'], // Force WebSocket only
  upgrade: false, // Prevent transport upgrades
  reconnectionAttempts: 5,
  timeout: 20000
};

const socket = io(WS_URL, socketOptions);
```

### 2. Fix Disconnect Handler in Backend

Ensure your disconnect handler accepts the reason parameter:

```python
@socketio.on('disconnect')
def handle_disconnect(reason):
    print(f'Client disconnected: {reason}')
```

### 3. Check for Path Mismatches

Ensure the Socket.IO path is consistent between frontend and backend:

- Frontend: `path: '/api/socket.io'` or `path: '/socket.io'`
- Backend: Match the path configuration accordingly

### 4. Increase Socket Timeout Settings

```python
# In production_server.py
socketio.server.eio.ping_interval = 25
socketio.server.eio.ping_timeout = 60
```

### 5. Check Proxy Headers

If using a reverse proxy, ensure it's properly configured for WebSockets:

```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "Upgrade";
```

### 6. Test Direct Connection

To isolate if it's a proxy issue, test connecting directly to the backend server when possible.

### 7. Verify CORS Settings

Make sure your CORS settings include all required origins and support WebSocket connections:

```python
# In your Flask app
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins=CORS_ORIGINS)
```

### 8. Check Network Logs

Use browser DevTools Network tab (filter by WS) to see if WebSocket connections are being attempted and what errors occur.

## Production URLs

After deployment, your application will be available at:
- **Frontend:** `http://your-server-ip:8080`
- **Backend API:** `http://your-server-ip:5000/api`

## Security Considerations

1. **Use HTTPS in production**
2. **Set strong Flask secret key**
3. **Restrict CORS origins to your domain only**
4. **Use environment variables for sensitive data**
5. **Keep API tokens secure**
6. **Regular security updates**

For questions or support, check the main README.md file.
