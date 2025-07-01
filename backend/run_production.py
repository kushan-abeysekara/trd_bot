import os
from dotenv import load_dotenv
from app import create_app

# Load production environment variables
load_dotenv('.env.production')

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '127.0.0.1')
    app.run(host=host, port=port, debug=False)
