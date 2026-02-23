"""
Cloud UBA Phase 6 - Report Generator
This file contains functions for generating comprehensive explanation reports.
"""

import logging
import pandas as pd
from datetime import datetime
from explainers.shap_explainer import explain_with_shap
from explainers.lime_explainer import explain_with_lime
from utils.preprocessing import extract_features_from_event

# Get logger
logger = logging.getLogger(__name__)

def generate_explanation(event_data, prediction_result):
    """
    Generate a comprehensive explanation for a prediction
    
    Args:
        event_data: Dictionary containing event data
        prediction_result: Dictionary containing prediction results
        
    Returns:
        Dictionary containing explanation
    """
    try:
        logger.info(f"Generating explanation for user {event_data.get('user_id')}")
        
        # Extract features from event data
        features = extract_features_from_event(event_data)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Generate explanations
        shap_explanation = explain_with_shap(features_df)
        lime_explanation = explain_with_lime(features_df)
        
        # Generate natural language explanation
        text_explanation = generate_text_explanation(
            event_data, 
            prediction_result, 
            shap_explanation
        )
        
        # Combine explanations
        explanation = {
            'user_id': event_data.get('user_id'),
            'timestamp': event_data.get('timestamp'),
            'prediction': prediction_result,
            'text_explanation': text_explanation,
            'shap_explanation': shap_explanation,
            'lime_explanation': lime_explanation,
            'explanation_time': datetime.utcnow().isoformat() + 'Z'
        }
        
        logger.info("Explanation generation complete")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        # Return minimal explanation on error
        return {
            'user_id': event_data.get('user_id'),
            'timestamp': event_data.get('timestamp'),
            'text_explanation': f"An error occurred while generating the explanation: {str(e)}",
            'explanation_time': datetime.utcnow().isoformat() + 'Z'
        }

def generate_text_explanation(event_data, prediction_result, shap_explanation):
    """
    Generate a natural language explanation for a prediction
    
    Args:
        event_data: Dictionary containing event data
        prediction_result: Dictionary containing prediction results
        shap_explanation: Dictionary containing SHAP explanation
        
    Returns:
        String containing natural language explanation
    """
    try:
        # Get top contributing features
        top_features = shap_explanation.get('feature_contributions', [])[:5]
        
        # Start with basic explanation
        if prediction_result.get('is_anomaly', False):
            explanation = f"This event was flagged as anomalous with a confidence of {prediction_result.get('confidence', 0):.2f}. "
        else:
            explanation = f"This event appears to be normal with a confidence of {prediction_result.get('confidence', 0):.2f}. "
        
        # Add information about the event
        explanation += f"The event was a {event_data.get('action_type', 'unknown')} action "
        explanation += f"performed by user {event_data.get('user_id', 'unknown')}. "
        
        # Add top contributing factors
        if top_features:
            explanation += "The top factors influencing this decision were:\n\n"
            
            for i, feature in enumerate(top_features):
                feature_name = feature.get('feature', '')
                shap_value = feature.get('shap_value', 0)
                feature_value = feature.get('feature_value', '')
                
                # Format feature name for readability
                readable_name = feature_name.replace('_', ' ').title()
                
                # Describe contribution
                if shap_value > 0:
                    direction = "increased"
                else:
                    direction = "decreased"
                
                explanation += f"{i+1}. {readable_name} ({feature_value}) {direction} the anomaly score by {abs(shap_value):.4f}\n"
        
        # Add recommendations
        explanation += "\nRecommendations:\n"
        
        if prediction_result.get('is_anomaly', False):
            explanation += "- Investigate this user's recent activities\n"
            explanation += "- Check for similar patterns across other users\n"
            explanation += "- Consider temporarily restricting access if suspicious\n"
        else:
            explanation += "- No immediate action required\n"
            explanation += "- Continue monitoring for changes in behavior\n"
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating text explanation: {str(e)}")
        return f"This event was {'flagged as anomalous' if prediction_result.get('is_anomaly', False) else 'classified as normal'}. Detailed explanation could not be generated due to an error."