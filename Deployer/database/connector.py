"""
Cloud UBA Phase 6 - Database Connector
This file contains functions for database connection handling.
"""

import sqlite3
import os
import logging
from flask import current_app, g
from database.models import create_tables, insert_sample_data

# Get logger
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Get a database connection
    
    Returns:
        SQLite connection object
    """
    try:
        # Check if connection exists in Flask application context
        if hasattr(g, 'db_conn'):
            return g.db_conn
        
        # Get database URI from config
        if current_app:
            database_uri = current_app.config.get('DATABASE_URI', 'sqlite:///cloud_uba.db')
        else:
            database_uri = os.getenv('DATABASE_URI', 'sqlite:///cloud_uba.db')
        
        # Extract path from URI
        if database_uri.startswith('sqlite:///'):
            db_path = database_uri[10:]
        else:
            db_path = 'cloud_uba.db'
        
        # Ensure directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        # Create connection
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        
        # Initialize database if needed
        create_tables(conn)
        
        # Insert sample data if in development mode
        if current_app and current_app.config.get('DEBUG', False):
            insert_sample_data(conn)
        
        # Store in Flask application context if available
        if current_app:
            g.db_conn = conn
        
        logger.debug(f"Database connection established to {db_path}")
        return conn
        
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def close_db_connection(e=None):
    """
    Close database connection
    
    Args:
        e: Exception (if any)
    """
    try:
        conn = g.pop('db_conn', None)
        
        if conn is not None:
            conn.close()
            logger.debug("Database connection closed")
            
    except Exception as e:
        logger.error(f"Error closing database connection: {str(e)}")

def init_app(app):
    """
    Initialize database with Flask application
    
    Args:
        app: Flask application
    """
    # Register close_db_connection to be called when application context ends
    app.teardown_appcontext(close_db_connection)
    
    # Initialize database on startup
    with app.app_context():
        get_db_connection()
        logger.info("Database initialized")