"""
Cloud UBA Phase 6 - Demonstration Module
This module provides demonstration capabilities for the XAI functionality.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

# Get logger
logger = logging.getLogger(__name__)

# Load necessary components
def load_demo_components():
    """
    Load demonstration components including model, preprocessor, and sample data
    
    Returns:
        Dictionary containing loaded components
    """
    try:
        # Define paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'random_forest_tuned.pkl')
        preprocessor_path = os.path.join(base_dir, 'models', 'preprocessor.pkl')
        feature_names_path = os.path.join(base_dir, 'data', 'processed', 'feature_names.pkl')
        
        # Sample data paths
        X_test_path = os.path.join(base_dir, 'data', 'processed', 'X_test.npy')
        y_test_path = os.path.join(base_dir, 'data', 'processed', 'y_test.npy')
        
        # Load model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        feature_names = joblib.load(feature_names_path)
        
        # Load sample test data
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        
        # Find example indices
        example_indices = {
            'true_positive': None,
            'true_negative': None,
            'false_positive': None,
            'false_negative': None
        }
        
        # Get predictions on test data
        y_pred = model.predict(X_test)
        
        # Find example indices for each case
        for i in range(len(y_test)):
            if y_test[i] == 1 and y_pred[i] == 1 and example_indices['true_positive'] is None:
                example_indices['true_positive'] = i
            elif y_test[i] == 0 and y_pred[i] == 0 and example_indices['true_negative'] is None:
                example_indices['true_negative'] = i
            elif y_test[i] == 0 and y_pred[i] == 1 and example_indices['false_positive'] is None:
                example_indices['false_positive'] = i
            elif y_test[i] == 1 and y_pred[i] == 0 and example_indices['false_negative'] is None:
                example_indices['false_negative'] = i
            
            # Break if we found all examples
            if all(v is not None for v in example_indices.values()):
                break
        
        # Try to load SHAP explainer, but provide fallback if not available
        explainer = None
        try:
            import shap
            explainer = shap.TreeExplainer(model)
        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            logger.warning(f"SHAP not available, using feature importance fallback: {str(e)}")
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'X_test': X_test,
            'y_test': y_test,
            'example_indices': example_indices,
            'explainer': explainer
        }
    
    except Exception as e:
        logger.error(f"Error loading demo components: {str(e)}")
        raise

def explain_prediction(instance_idx, components):
    """
    Generate explanation for a prediction
    
    Args:
        instance_idx: Index of the instance to explain
        components: Dictionary of loaded components
        
    Returns:
        Dictionary containing explanation
    """
    try:
        # Extract components
        model = components['model']
        X_test = components['X_test']
        y_test = components['y_test']
        feature_names = components['feature_names']
        explainer = components['explainer']
        
        # Get instance
        instance = X_test[instance_idx].reshape(1, -1)
        
        # Get prediction and probability
        prediction = model.predict(instance)[0]
        proba = model.predict_proba(instance)[0]
        
        # Get feature contributions - either using SHAP or feature importance
        feature_contributions = []
        
        if explainer is not None:
            # Use SHAP if available
            try:
                shap_values = explainer.shap_values(instance)[1][0]  # For class 1 (anomaly)
                
                # Create feature contributions
                for i, (name, value) in enumerate(zip(feature_names, shap_values)):
                    feature_contributions.append({
                        'feature': name,
                        'shap_value': float(value),
                        'feature_value': float(instance[0, i])
                    })
            except Exception as e:
                logger.warning(f"SHAP explanation failed, using feature importance fallback: {str(e)}")
                explainer = None  # Force fallback
        
        # Fallback to feature importance if SHAP is not available
        if explainer is None:
            # Get feature importances from the model
            importances = model.feature_importances_
            
            # Create feature contributions based on feature importance
            for i, (name, importance) in enumerate(zip(feature_names, importances)):
                # Direction is based on whether the feature value is above the mean
                direction = 1 if instance[0, i] > np.mean(X_test[:, i]) else -1
                feature_contributions.append({
                    'feature': name,
                    'shap_value': float(importance * direction),  # Approximate direction
                    'feature_value': float(instance[0, i])
                })
        
        # Sort by absolute contribution value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Generate explanation text
        explanation_text = "This event was classified as "
        explanation_text += "an anomaly" if prediction == 1 else "normal"
        explanation_text += f" with {proba[prediction] * 100:.1f}% confidence.\n\n"
        
        explanation_text += "The most important factors in this decision were:\n"
        
        for i, contrib in enumerate(feature_contributions[:5]):
            feature = contrib['feature']
            contribution = contrib['shap_value']
            feature_value = contrib['feature_value']
            
            # Format the explanation based on the feature
            if contribution > 0:
                explanation_text += f"{i+1}. {feature} = {feature_value:.2f} increased the anomaly score\n"
            else:
                explanation_text += f"{i+1}. {feature} = {feature_value:.2f} decreased the anomaly score\n"
        
        # Create result
        result = {
            'predicted_label': 'Anomaly' if prediction == 1 else 'Normal',
            'true_label': 'Anomaly' if y_test[instance_idx] == 1 else 'Normal',
            'confidence': float(proba[prediction]),
            'feature_contributions': feature_contributions,
            'explanation_text': explanation_text
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error explaining prediction: {str(e)}")
        raise

def generate_analyst_report(instance_idx, components, df=None):
    """
    Generate a security analyst report for a specific instance.
    
    Args:
        instance_idx: Index of the instance to explain
        components: Dictionary of loaded components
        df: DataFrame containing raw data (optional)
        
    Returns:
        Formatted report text
    """
    # Get explanation
    explanation = explain_prediction(instance_idx, components)

    # Create report
    report = "=" * 80 + "\n"
    report += "CLOUD USER BEHAVIOR ANALYTICS - SECURITY ALERT REPORT\n"
    report += "=" * 80 + "\n\n"

    # Basic information
    report += f"Alert ID: ALT-{instance_idx:06d}\n"
    report += f"Classification: {explanation['predicted_label']}\n"
    report += f"Confidence: {explanation['confidence']:.2f}\n"
    report += f"True Label (for evaluation): {explanation['true_label']}\n\n"

    # User activity details
    if df is not None and instance_idx < len(df):
        user_id = df.iloc[instance_idx]['user_id']
        action = df.iloc[instance_idx]['action_type']
        hour = df.iloc[instance_idx]['hour_of_day']
        day = df.iloc[instance_idx]['day_of_week']
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day]

        report += f"User ID: {user_id}\n"
        report += f"Action: {action}\n"
        report += f"Time: {hour:02d}:00 on {day_name}\n"
        report += f"Failed Attempts: {df.iloc[instance_idx]['failed_attempts']}\n"
        report += f"Data Volume: {df.iloc[instance_idx]['data_volume_mb']:.2f} MB\n"
        report += f"Location Change: {'Yes' if df.iloc[instance_idx]['location_change'] == 1 else 'No'}\n\n"
    else:
        report += "User Activity Details: Not available for this instance\n\n"

    # Explanation
    report += "EXPLANATION\n"
    report += "-" * 80 + "\n"
    report += explanation['explanation_text'] + "\n\n"

    # Recommendations
    report += "RECOMMENDED ACTIONS\n"
    report += "-" * 80 + "\n"

    if explanation['predicted_label'] == 'Anomaly':
        report += "1. Investigate this user's recent activities for additional suspicious patterns\n"
        report += "2. Consider temporarily restricting access privileges pending investigation\n"
        report += "3. Check for similar patterns across other users\n"

        # Add specific recommendations based on features
        for contrib in explanation['feature_contributions'][:5]:
            if contrib['shap_value'] > 0:
                feature = contrib['feature']
                if 'failed' in feature.lower():
                    report += "4. Review authentication logs for brute force attempts\n"
                elif 'admin' in feature.lower():
                    report += "4. Verify if administrative actions were authorized\n"
                elif 'night' in feature.lower() or 'hour' in feature.lower():
                    report += "4. Confirm if after-hours activity is expected for this user\n"
                elif 'data' in feature.lower() and 'volume' in feature.lower():
                    report += "4. Check data exfiltration monitoring systems\n"
                elif 'location' in feature.lower():
                    report += "4. Verify if location change is consistent with user's travel schedule\n"
    else:
        report += "1. No immediate action required\n"
        report += "2. Consider adding to baseline of normal behavior\n"

    return report

def get_demo_examples():
    """
    Get demonstration examples
    
    Returns:
        Dictionary containing example data
    """
    try:
        # Load components
        components = load_demo_components()
        
        # Get example indices
        example_indices = components['example_indices']
        
        # Generate explanations for each example
        examples = {}
        for case_name, idx in example_indices.items():
            if idx is not None:
                explanation = explain_prediction(idx, components)
                report = generate_analyst_report(idx, components)
                
                examples[case_name] = {
                    'index': int(idx),
                    'explanation': explanation,
                    'report': report
                }
        
        return examples
    
    except Exception as e:
        logger.error(f"Error getting demo examples: {str(e)}")
        raise