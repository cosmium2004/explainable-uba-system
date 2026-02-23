"""
Cloud UBA Phase 6 - API Middleware
This file contains middleware functions for API routes.
"""

import logging
from functools import wraps
from flask import request, jsonify, current_app

# Get logger
logger = logging.getLogger(__name__)

def token_required(f):
    """
    Decorator for routes that require authentication
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Get token from header
        if 'X-API-Key' in request.headers:
            token = request.headers['X-API-Key']
        
        # Check if token is valid
        if not token:
            logger.warning("API request missing token")
            return jsonify({
                'success': False,
                'error': 'Authentication failed',
                'message': 'API key is missing'
            }), 401
        
        # Validate token
        valid_tokens = current_app.config.get('API_KEYS', [])
        if token not in valid_tokens:
            logger.warning(f"Invalid API key: {token[:5]}...")
            return jsonify({
                'success': False,
                'error': 'Authentication failed',
                'message': 'Invalid API key'
            }), 401
        
        logger.debug("API key validated successfully")
        return f(*args, **kwargs)
    
    return decorated

def validate_input(validator_func):
    """
    Decorator for routes that require input validation
    
    Args:
        validator_func: Function to validate input
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Get data from request
            data = request.get_json()
            
            if not data:
                logger.warning("API request missing data")
                return jsonify({
                    'success': False,
                    'error': 'Validation failed',
                    'message': 'No data provided'
                }), 400
            
            # Validate data
            is_valid, error = validator_func(data)
            
            if not is_valid:
                logger.warning(f"Validation failed: {error}")
                return jsonify({
                    'success': False,
                    'error': 'Validation failed',
                    'message': error
                }), 400
            
            logger.debug("Input validation passed")
            return f(*args, **kwargs)
        
        return decorated
    
    return decorator

def log_request():
    """
    Log request details
    """
    logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")

def log_response(response):
    """
    Log response details
    
    Args:
        response: Flask response object
    
    Returns:
        The response object
    """
    logger.info(f"Response: {response.status_code}")
    return response