"""
Cloud UBA Phase 6 - Prediction Logic
This file contains functions for making predictions using the trained model.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import joblib
from models.model_loader import get_model, get_feature_pipeline

# Get logger
logger = logging.getLogger(__name__)

# Cache for user stats and global stats
_user_stats = None
_global_stats = None

def _get_user_stats():
    """Load cached user statistics from training data"""
    global _user_stats
    if _user_stats is None:
        stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_stats.pkl')
        if os.path.exists(stats_path):
            _user_stats = joblib.load(stats_path)
            logger.info("User stats loaded successfully")
        else:
            logger.warning("User stats file not found, using defaults")
            _user_stats = pd.DataFrame()
    return _user_stats

def _get_global_stats():
    """Load cached global statistics from training data"""
    global _global_stats
    if _global_stats is None:
        stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_stats.pkl')
        if os.path.exists(stats_path):
            _global_stats = joblib.load(stats_path)
            logger.info("Global stats loaded successfully")
        else:
            logger.warning("Global stats file not found, using defaults")
            _global_stats = {
                'resource_access_count': {'mean': 50.0, 'std': 30.0},
                'failed_attempts': {'mean': 1.0, 'std': 2.0},
                'data_volume_mb': {'mean': 500.0, 'std': 300.0},
                'data_volume_mb_median': 400.0
            }
    return _global_stats

def _engineer_features(event_data):
    """
    Replicate the notebook's feature engineering pipeline.
    
    Takes raw API event_data and produces a DataFrame with the columns the
    ColumnTransformer preprocessor expects:
      Categorical: action_type, time_of_day
      Numerical: hour_of_day, day_of_week, resource_access_count, failed_attempts,
                 data_volume_mb, location_change, is_weekend,
                 resource_access_count_mean, resource_access_count_std,
                 failed_attempts_mean, failed_attempts_sum,
                 data_volume_mb_mean, data_volume_mb_max,
                 location_change_mean,
                 resource_access_count_zscore, failed_attempts_zscore, data_volume_mb_zscore
    """
    # Extract basic fields from the API request
    hour_of_day = int(event_data.get('hour_of_day', 12))
    day_of_week = int(event_data.get('day_of_week', 0))
    action_type = event_data.get('action_type', 'login')
    resource_access_count = int(event_data.get('resource_access_count', 1))
    failed_attempts = int(event_data.get('failed_attempts', 0))
    data_volume_mb = float(event_data.get('data_volume_mb', 0.0))
    location_change = int(event_data.get('location_change', 0))
    
    # If timestamp is provided but hour/day aren't, derive them
    if 'timestamp' in event_data and 'hour_of_day' not in event_data:
        try:
            ts = event_data['timestamp'].replace('Z', '+00:00')
            dt = datetime.fromisoformat(ts)
            hour_of_day = dt.hour
            day_of_week = dt.weekday()
        except Exception:
            pass
    
    # 1. Time-based features
    if 5 <= hour_of_day < 12:
        time_of_day = 'morning'
    elif 12 <= hour_of_day < 17:
        time_of_day = 'afternoon'
    elif 17 <= hour_of_day < 22:
        time_of_day = 'evening'
    else:
        time_of_day = 'night'
    
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # 2. User behavior stats (lookup from training data)
    user_stats = _get_user_stats()
    user_id = event_data.get('user_id', '')
    
    # Try to find user stats; use global averages as defaults
    try:
        user_id_int = int(user_id) if str(user_id).replace('-', '').isdigit() else -1
    except (ValueError, TypeError):
        user_id_int = -1
    
    if not user_stats.empty and user_id_int in user_stats['user_id'].values:
        row = user_stats[user_stats['user_id'] == user_id_int].iloc[0]
        rac_mean = float(row['resource_access_count_mean'])
        rac_std = float(row['resource_access_count_std'])
        fa_mean = float(row['failed_attempts_mean'])
        fa_sum = float(row['failed_attempts_sum'])
        dv_mean = float(row['data_volume_mb_mean'])
        dv_max = float(row['data_volume_mb_max'])
        lc_mean = float(row['location_change_mean'])
    else:
        # Use reasonable defaults from the training distribution
        rac_mean = 50.0
        rac_std = 30.0
        fa_mean = 1.0
        fa_sum = 100.0
        dv_mean = 500.0
        dv_max = 1000.0
        lc_mean = 0.3
    
    # 3. Z-scores using global training statistics
    global_stats = _get_global_stats()
    rac_zscore = (resource_access_count - global_stats['resource_access_count']['mean']) / max(global_stats['resource_access_count']['std'], 1e-10)
    fa_zscore = (failed_attempts - global_stats['failed_attempts']['mean']) / max(global_stats['failed_attempts']['std'], 1e-10)
    dv_zscore = (data_volume_mb - global_stats['data_volume_mb']['mean']) / max(global_stats['data_volume_mb']['std'], 1e-10)
    
    # Build the feature row (must match the column order the preprocessor expects)
    feature_row = {
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        'resource_access_count': resource_access_count,
        'failed_attempts': failed_attempts,
        'data_volume_mb': data_volume_mb,
        'location_change': location_change,
        'is_weekend': is_weekend,
        'resource_access_count_mean': rac_mean,
        'resource_access_count_std': rac_std,
        'failed_attempts_mean': fa_mean,
        'failed_attempts_sum': fa_sum,
        'data_volume_mb_mean': dv_mean,
        'data_volume_mb_max': dv_max,
        'location_change_mean': lc_mean,
        'resource_access_count_zscore': rac_zscore,
        'failed_attempts_zscore': fa_zscore,
        'data_volume_mb_zscore': dv_zscore,
        'action_type': action_type,
        'time_of_day': time_of_day,
    }
    
    return pd.DataFrame([feature_row])


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

        # Get model and feature pipeline (ColumnTransformer)
        model = get_model()
        pipeline = get_feature_pipeline()

        # Engineer features to match training pipeline (26 features)
        features_df = _engineer_features(event_data)

        # Transform features using ColumnTransformer (StandardScaler + OneHotEncoder)
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