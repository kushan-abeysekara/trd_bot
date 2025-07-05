#!/bin/bash

# Production deployment script for DigitalOcean

echo "🚀 Setting up production environment..."

# Create production .env file
cat > .env << EOF
# Production Environment Variables
FLASK_ENV=production
FLASK_DEBUG=False
DERIV_APP_ID=1089
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000,https://tradingbot-4iuxi.ondigitalocean.app
HOST=0.0.0.0
PORT=5000
EOF

echo "✅ Production .env file created"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🌐 CORS configured for:"
echo "  - http://localhost:3000 (local development)"
echo "  - https://tradingbot-4iuxi.ondigitalocean.app (production)"

echo "🚀 Ready to deploy!"
echo "Run with: python app.py"
