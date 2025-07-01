#!/bin/bash

echo "Starting deployment..."

# Build and deploy backend
echo "Building backend Docker image..."
cd backend
docker build -t tradingbot-backend .

echo "Running backend container..."
docker run -d \
  --name tradingbot-backend \
  -p 8080:8080 \
  --env-file .env.production \
  tradingbot-backend

echo "Waiting for backend to start..."
sleep 30

# Test health check
echo "Testing health check..."
curl -f http://localhost:8080/api/health || {
  echo "Health check failed!"
  docker logs tradingbot-backend
  exit 1
}

echo "Backend deployed successfully!"

# Build and deploy frontend
echo "Building frontend..."
cd ../frontend
npm run build

echo "Deployment complete!"
