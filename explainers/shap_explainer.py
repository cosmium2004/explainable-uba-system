"""
Cloud UBA Phase 6 - SHAP Explainer
This file contains functions for generating SHAP-based explanations.
"""

import shap
import numpy as np
import pandas as pd
import logging
from models.model_loader import get_model, get_shap_explainer

# Get logger
logger = logging.getLogger(__name__)

def explain_with_shap(features, feature_names=None):
    """
    Generate SHAP-based explanation for a prediction
    
    Args:
        features: Feature vector or DataFrame
        feature_names: List of feature names (optional)
        
    Returns:
        Dictionary containing SHAP explanation
    """
    try:
        logger.info("Generating SHAP explanation")
        
        # Convert features to DataFrame if needed
        if not isinstance(features, pd.DataFrame):
            if feature_names is None:
                raise ValueError("Feature names must be provided for non-DataFrame input")
            features = pd.DataFrame([features], columns=feature_names)
        
        # Get SHAP explainer
        try:
            explainer = get_shap_explainer()
        except Exception as e:
            logger.warning(f"Could not load cached SHAP explainer: {e}")
            # Create a new explainer if loading fails
            model = get_model()
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(features)
        
        # For binary classification, we're interested in class 1 (anomaly)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]
        
        # Get base value (expected value)
        if hasattr(explainer, 'expected_value'):
            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[1]  # Class 1 (anomaly)
            else:
                base_value = explainer.expected_value
        else:
            base_value = 0.5  # Default if not available
        
        # Create feature contribution list
        feature_contributions = []
        for i, name in enumerate(features.columns):
            contribution = {
                'feature': name,
                'shap_value': float(shap_values[0][i]),
                'feature_value': float(features.iloc[0, i]) if np.issubdtype(features.dtypes[i], np.number) else str(features.iloc[0, i])
            }
            feature_contributions.append(contribution)
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Create explanation
        explanation = {
            'base_value': float(base_value),
            'output_value': float(base_value + np.sum(shap_values)),
            'feature_contributions': feature_contributions
        }
        
        logger.info(f"SHAP explanation generated with {len(feature_contributions)} features")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanation: {str(e)}")
        # Return minimal explanation on error
        return {
            'base_value': 0.5,
            'output_value': 0.5,
            'feature_contributions': [],
            'error': str(e)
        }