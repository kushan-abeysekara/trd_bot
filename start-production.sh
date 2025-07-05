#!/bin/bash

echo "🚀 Starting Deriv Trading Bot - Production Mode..."
echo ""

# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=False

echo "📦 Installing/updating Python dependencies..."
cd backend
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install backend dependencies!"
    exit 1
fi

echo "📦 Installing/updating Node.js dependencies..."
cd ../frontend
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies!"
    exit 1
fi

echo "🏗️ Building frontend for production..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Failed to build frontend!"
    exit 1
fi

echo ""
echo "✅ Production build complete!"
echo ""
echo "🚀 Starting production servers..."
echo ""

# Start backend server in production mode
echo "Starting backend server..."
cd ../backend
FLASK_ENV=production python production_server.py &
BACKEND_PID=$!

sleep 3

# Start frontend production server
echo "Starting frontend production server..."
cd ../frontend
npm run start-production &
FRONTEND_PID=$!

echo ""
echo "✅ Production servers are running!"
echo ""
echo "🌐 Access your application at:"
echo "Frontend: http://your-server-ip:8080"
echo "Backend API: http://your-server-ip:5000/api"
echo ""
echo "For external access, make sure ports 5000 and 8080 are open in your firewall."
echo ""
echo "To stop the servers, run: kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
