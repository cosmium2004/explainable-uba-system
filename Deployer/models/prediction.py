"""
Cloud UBA Phase 6 - Prediction Logic
This file contains functions for making predictions using the trained model.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import logging
from models.model_loader import get_model, get_feature_pipeline
from utils.preprocessing import extract_features_from_event

# Get logger
logger = logging.getLogger(__name__)

def predict_anomaly(event_data):
    """
    Make a prediction for a single event
    Args:
        event_data: Dictionary containing event data
    Returns:
        Dictionary with prediction results
    """
    try:
        logger.info(f"Making prediction for user {event_data.get('user_id')}")

        # Use the detailed feature extraction function
        from utils.preprocessing import extract_features_from_event
        features = extract_features_from_event(event_data)

        # Get model and feature pipeline
        model = get_model()
        pipeline = get_feature_pipeline()

        # Transform features using pipeline
        features_df = pd.DataFrame([features])
        transformed_features = pipeline.transform(features_df)

        # Make prediction
        prediction_proba = model.predict_proba(transformed_features)[0]
        prediction = model.predict(transformed_features)[0]

        # Get anomaly score (probability of class 1)
        anomaly_score = float(prediction_proba[1])

        # Create result
        result = {
            'user_id': event_data.get('user_id'),
            'timestamp': event_data.get('timestamp'),
            'is_anomaly': bool(prediction == 1),
            'anomaly_score': anomaly_score,
            'prediction_time': datetime.utcnow().isoformat() + 'Z',
            'confidence': anomaly_score if prediction == 1 else 1 - anomaly_score
        }

        logger.info(f"Prediction complete: {'Anomaly' if result['is_anomaly'] else 'Normal'} "
                   f"with score {result['anomaly_score']:.4f}")
        return result
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise
        
def batch_predict(events):
    """
    Make predictions for multiple events
    
    Args:
        events: List of dictionaries containing event data
        
    Returns:
        List of dictionaries with prediction results
    """
    try:
        logger.info(f"Processing batch prediction for {len(events)} events")
        
        results = []
        for event in events:
            result = predict_anomaly(event)
            results.append(result)
            
        logger.info(f"Batch prediction complete. Found {sum(1 for r in results if r['is_anomaly'])} anomalies")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise
        
def get_threshold():
    """
    Get the current anomaly threshold
    
    Returns:
        Float representing the threshold
    """
    # This could be loaded from a configuration or database
    # For now, we'll use a fixed value
    return 0.7