module.exports = {
  apps: [{
    name: 'tradingbot-backend',
    script: 'venv/bin/python',
    args: 'run_production.py',
    cwd: '/var/www/tradingbot/backend',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      PYTHONPATH: '/var/www/tradingbot/backend'
    },
    error_file: '/var/log/tradingbot/backend-error.log',
    out_file: '/var/log/tradingbot/backend-out.log',
    log_file: '/var/log/tradingbot/backend-combined.log'
  }]
};
