"""
Cloud UBA Phase 6 - LIME Explainer
This file contains functions for generating LIME-based explanations.
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import logging
from models.model_loader import get_model, get_feature_pipeline

# Get logger
logger = logging.getLogger(__name__)

# Cache for LIME explainer
_lime_explainer = None

def get_lime_explainer(feature_names, categorical_features=None):
    """
    Get or create a LIME explainer
    
    Args:
        feature_names: List of feature names
        categorical_features: List of indices of categorical features
        
    Returns:
        LIME explainer
    """
    global _lime_explainer
    
    if _lime_explainer is None:
        logger.info("Creating new LIME explainer")
        
        # If categorical_features not provided, try to infer
        if categorical_features is None:
            categorical_features = []
            for i, name in enumerate(feature_names):
                # Heuristic: features with these prefixes are likely categorical
                if any(name.startswith(prefix) for prefix in 
                      ['action_type_', 'device_type_', 'resource_type_', 'is_', 'time_of_day']):
                    categorical_features.append(i)
        
        # Create explainer
        _lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.zeros((1, len(feature_names))),  # Dummy data, will be updated later
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_names=['normal', 'anomaly'],
            mode='classification'
        )
    
    return _lime_explainer

def explain_with_lime(features, feature_names=None):
    """
    Generate LIME-based explanation for a prediction
    
    Args:
        features: Feature vector or DataFrame
        feature_names: List of feature names (optional)
        
    Returns:
        Dictionary containing LIME explanation
    """
    try:
        logger.info("Generating LIME explanation")
        
        # Convert features to DataFrame if needed
        if not isinstance(features, pd.DataFrame):
            if feature_names is None:
                raise ValueError("Feature names must be provided for non-DataFrame input")
            features = pd.DataFrame([features], columns=feature_names)
        
        # Get model and feature names
        model = get_model()
        feature_names = list(features.columns)
        
        # Determine categorical features
        categorical_features = []
        for i, name in enumerate(feature_names):
            if any(name.startswith(prefix) for prefix in 
                  ['action_type_', 'device_type_', 'resource_type_', 'is_', 'time_of_day']):
                categorical_features.append(i)
        
        # Get LIME explainer
        explainer = get_lime_explainer(feature_names, categorical_features)
        
        # Define prediction function for LIME
        def predict_fn(x):
            return model.predict_proba(x)
        
        # Generate explanation
        lime_exp = explainer.explain_instance(
            features.values[0], 
            predict_fn,
            num_features=10,
            top_labels=1
        )
        
        # Extract explanation for anomaly class (class 1)
        class_idx = 1  # Anomaly class
        lime_features = lime_exp.as_list(label=class_idx)
        
        # Create feature contribution list
        feature_contributions = []
        for feature_desc, contribution in lime_features:
            # Parse feature name from LIME's description
            parts = feature_desc.split(' ')
            feature_name = parts[0]
            
            # Add to list
            feature_contributions.append({
                'feature': feature_name,
                'contribution': float(contribution),
                'description': feature_desc
            })
        
        # Create explanation
        explanation = {
            'feature_contributions': feature_contributions,
            'intercept': float(lime_exp.intercept[class_idx]),
            'score': float(lime_exp.score)
        }
        
        logger.info(f"LIME explanation generated with {len(feature_contributions)} features")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating LIME explanation: {str(e)}")
        # Return minimal explanation on error
        return {
            'feature_contributions': [],
            'intercept': 0.0,
            'score': 0.0,
            'error': str(e)
        }