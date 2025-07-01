import os
from dotenv import load_dotenv
from app import create_app

# Load production environment variables
load_dotenv('.env.production')

app = create_app()

if __name__ == '__main__':
    # Use port 8080 for deployment platform compatibility
    port = int(os.getenv('PORT', 8080))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
