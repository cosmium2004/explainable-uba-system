"""
Cloud UBA Phase 6 - Configuration Settings
This file contains configuration settings for different environments.
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    VERSION = '1.0.0'
    SECRET_KEY = os.getenv('SECRET_KEY', 'cloud-uba-secret-key')
    DEBUG = False
    TESTING = False
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    RF_MODEL_PATH = os.path.join(MODEL_PATH, 'rf_model.pkl')
    FEATURE_PIPELINE_PATH = os.path.join(MODEL_PATH, 'feature_pipeline.pkl')
    
    # Explanation settings
    SHAP_EXPLAINER_PATH = os.path.join(MODEL_PATH, 'shap_explainer.pkl')
    
    # Database settings
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///cloud_uba.db')
    
    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    
    # Monitoring settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/cloud_uba.log')
    
    # Alert settings
    ALERT_THRESHOLD = float(os.getenv('ALERT_THRESHOLD', '0.8'))
    
    # API rate limiting
    RATE_LIMIT = os.getenv('RATE_LIMIT', '100/hour')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URI = 'sqlite:///test.db'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # In production, these should be set via environment variables
    SECRET_KEY = os.getenv('SECRET_KEY')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')
    DATABASE_URI = os.getenv('DATABASE_URI')


# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig
}