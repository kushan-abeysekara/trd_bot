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

## WebSocket Configuration (Important!)

WebSockets require special configuration to work correctly in production:

### 1. Nginx WebSocket Configuration

Ensure your Nginx configuration includes proper WebSocket proxy settings:

```nginx
# WebSocket for Socket.IO
location /socket.io/ {
    proxy_pass http://localhost:5000/socket.io/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    
    # WebSocket timeouts
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}

# Additional path if using API prefix
location /api/socket.io/ {
    proxy_pass http://localhost:5000/socket.io/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "Upgrade";
    proxy_set_header Host $host;
    
    # WebSocket timeouts
    proxy_read_timeout 3600s;
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
