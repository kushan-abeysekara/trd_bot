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

- `SECRET_KEY`: Your secret key for production
- `DATABASE_URL`: Your database connection string
- `REDIS_URL`: Your Redis connection string (if used)
- `SOCKETIO_MESSAGE_QUEUE`: Set to `redis` if using Redis, otherwise `none`
- `API_CORS_ORIGINS`: Set to `*` to allow all origins or specify your frontend URL

#### Frontend Service:

- `REACT_APP_API_URL`: Set to your backend API URL
- `REACT_APP_SOCKET_IO_URL`: Set to your Socket.IO URL

### 4. Build and Deployment

After configuring the above settings:

1. Push your code to the GitHub repository.
2. Trigger a deployment in the DigitalOcean App Platform dashboard.
3. Monitor the deployment logs for any errors.
4. Once deployed, test the application thoroughly to ensure all features work as expected.

### 5. Troubleshooting Common Issues

- **WebSocket connection errors**: Double-check your Socket.IO route configuration and ensure your backend is running.
- **CORS issues**: Ensure `API_CORS_ORIGINS` is set correctly in your environment variables.
- **Dependency errors**: Verify your `requirements.txt` and `package.json` files are up-to-date and include all necessary dependencies.

For further assistance, refer to the [DigitalOcean App Platform documentation](https://www.digitalocean.com/docs/app-platform/) or seek help from the community forums.
