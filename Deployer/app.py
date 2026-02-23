"""
Cloud UBA Phase 6 - Main Application
This file contains the main Flask application.
"""

import os
import logging
from flask import Flask, jsonify, render_template, request
from api.routes import api_blueprint
from database.connector import init_app as init_db
from monitoring.logger import setup_logging

def create_app(test_config=None):
    """
    Create and configure the Flask application
    
    Args:
        test_config: Test configuration (optional)
        
    Returns:
        Flask application
    """
    # Set up logging
    setup_logging()
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info("Creating Flask application")
    
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Set default configuration
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev'),
        DATABASE_URI=os.getenv('DATABASE_URI', 'sqlite:///instance/cloud_uba.db'),
        API_KEYS=[os.getenv('API_KEY', 'test-api-key')],
        DEBUG=os.getenv('DEBUG', 'False').lower() in ('true', '1', 't'),
        ALERT_THRESHOLD=float(os.getenv('ALERT_THRESHOLD', '0.8')),
        EMAIL_NOTIFICATIONS_ENABLED=os.getenv('EMAIL_NOTIFICATIONS_ENABLED', 'False').lower() in ('true', '1', 't'),
        SMTP_SERVER=os.getenv('SMTP_SERVER', 'smtp.example.com'),
        SMTP_PORT=int(os.getenv('SMTP_PORT', '587')),
        SMTP_USERNAME=os.getenv('SMTP_USERNAME', ''),
        SMTP_PASSWORD=os.getenv('SMTP_PASSWORD', ''),
        SENDER_EMAIL=os.getenv('SENDER_EMAIL', 'alerts@example.com'),
        RECIPIENT_EMAILS=os.getenv('RECIPIENT_EMAILS', '').split(',')
    )
    
    # Load test config if provided
    if test_config is not None:
        app.config.from_mapping(test_config)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Initialize database
    init_db(app)
    
    # Register blueprints
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    # Add health check route
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'ok',
            'version': '1.0.0'
        })
    
    # Add web interface routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/predict')
    def predict_page():
        # Pass the default API key to the template
        api_key = app.config['API_KEYS'][0]
        return render_template('predict.html', api_key=api_key)
    
    @app.route('/batch')
    def batch_page():
        api_key = app.config['API_KEYS'][0]
        return render_template('batch.html', api_key=api_key)
    
    @app.route('/dashboard')
    def dashboard_page():
        # In a real application, you would fetch data from your database here
        return render_template('dashboard.html')
    
    logger.info("Flask application created successfully")
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))