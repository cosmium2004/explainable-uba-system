"""
Cloud UBA Phase 6 - WSGI Entry Point
This file serves as the WSGI entry point for production deployment.
"""

import os
from app import app

if __name__ == "__main__":
    # Set environment to production
    os.environ['FLASK_ENV'] = 'production'
    
    # Run the application
    app.run()