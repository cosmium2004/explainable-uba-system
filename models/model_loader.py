"""
Cloud UBA Phase 6 - Model Loader
This file contains functions for loading and caching trained models.
"""

import os
import pickle
import logging
import joblib
import numpy as np
from datetime import datetime
import shap
from database.queries import get_model_version

# Get logger
logger = logging.getLogger(__name__)

# Cache for loaded models and pipelines
_model_cache = {}
_pipeline_cache = {}
_shap_explainer_cache = None

def get_model():
    """
    Get the trained model
    
    Returns:
        Trained model object
    """
    global _model_cache
    
    try:
        # Check if model is already loaded
        if 'model' in _model_cache:
            logger.debug("Using cached model")
            return _model_cache['model']
        
        # Get active model version from database
        model_version = get_model_version()
        
        if model_version:
            model_path = model_version['model_path']
            logger.info(f"Loading model version {model_version['version']} from {model_path}")
        else:
            # Use default model path
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            model_path = "D:/MiniProject/LLM/models/random_forest_tuned.pkl"
            logger.info(f"No model version found in database, using default model at {model_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            # Create a dummy model for testing
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10)
            model.fit(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), [0, 1, 1, 1])
            logger.warning("Created dummy model for testing")
        else:
            # Load model from file
            try:
                model = joblib.load(model_path)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model with joblib: {str(e)}")
                # Try with pickle as fallback
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Model loaded successfully with pickle")
        
        # Cache model
        _model_cache['model'] = model
        _model_cache['loaded_at'] = datetime.utcnow().isoformat()
        
        return model
        
    except Exception as e:
        logger.error(f"Error getting model: {str(e)}")
        # Create a dummy model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10)
        model.fit(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), [0, 1, 1, 1])
        logger.warning("Created dummy model for testing due to error")
        return model

def get_feature_pipeline():
    """
    Get the feature preprocessing pipeline
    
    Returns:
        Feature preprocessing pipeline
    """
    global _pipeline_cache
    
    try:
        # Check if pipeline is already loaded
        if 'pipeline' in _pipeline_cache:
            logger.debug("Using cached feature pipeline")
            return _pipeline_cache['pipeline']
        
        # Get model directory
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        pipeline_path = "D:/MiniProject/LLM/models/preprocessor.pkl"
        
        # Check if pipeline file exists
        if not os.path.exists(pipeline_path):
            logger.warning(f"Feature pipeline file not found: {pipeline_path}")
            # Create a dummy pipeline for testing
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([('scaler', StandardScaler())])
            logger.warning("Created dummy feature pipeline for testing")
        else:
            # Load pipeline from file
            try:
                pipeline = joblib.load(pipeline_path)
                logger.info("Feature pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Error loading pipeline with joblib: {str(e)}")
                # Try with pickle as fallback
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                logger.info("Feature pipeline loaded successfully with pickle")
        
        # Cache pipeline
        _pipeline_cache['pipeline'] = pipeline
        _pipeline_cache['loaded_at'] = datetime.utcnow().isoformat()
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error getting feature pipeline: {str(e)}")
        # Create a dummy pipeline for testing
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([('scaler', StandardScaler())])
        logger.warning("Created dummy feature pipeline for testing due to error")
        return pipeline

def get_shap_explainer():
    """
    Get the SHAP explainer
    
    Returns:
        SHAP explainer object
    """
    global _shap_explainer_cache
    
    try:
        # Check if explainer is already created
        if _shap_explainer_cache is not None:
            logger.debug("Using cached SHAP explainer")
            return _shap_explainer_cache
        
        # Get model
        model = get_model()
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer created successfully")
        
        # Cache explainer
        _shap_explainer_cache = explainer
        
        return explainer
        
    except Exception as e:
        logger.error(f"Error getting SHAP explainer: {str(e)}")
        raise

def reload_model():
    """
    Force reload of model and pipelines
    
    Returns:
        Boolean indicating success
    """
    global _model_cache, _pipeline_cache, _shap_explainer_cache
    
    try:
        # Clear caches
        _model_cache = {}
        _pipeline_cache = {}
        _shap_explainer_cache = None
        
        # Reload model and pipeline
        get_model()
        get_feature_pipeline()
        
        logger.info("Model and pipelines reloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        return False