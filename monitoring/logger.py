"""
Cloud UBA Phase 6 - Logging Configuration
This file contains functions for setting up logging.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from flask import current_app
import sys

def setup_logging():
    """
    Set up logging configuration
    """
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Set up root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    root_logger.setLevel(numeric_level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create file handler
    log_file = os.path.join(log_dir, 'cloud_uba.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Create loggers for different components
    create_component_logger('api', numeric_level)
    create_component_logger('models', numeric_level)
    create_component_logger('explainers', numeric_level)
    create_component_logger('monitoring', numeric_level)
    
    logging.info("Logging setup complete")

def create_component_logger(name, level):
    """
    Create a logger for a specific component
    
    Args:
        name: Component name
        level: Log level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Don't propagate to root logger to avoid duplicate logs
    logger.propagate = True