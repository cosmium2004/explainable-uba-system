"""
Cloud UBA Phase 6 - Performance Tracker
This file contains functions for tracking model performance.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from flask import current_app
from database.connector import get_db_connection

# Get logger
logger = logging.getLogger(__name__)

# In-memory cache for recent predictions
_recent_predictions = []
_performance_metrics = {
    'total_predictions': 0,
    'anomalies_detected': 0,
    'last_updated': datetime.utcnow().isoformat()
}

def log_prediction(event_data, prediction_result):
    """
    Log a prediction for monitoring
    
    Args:
        event_data: Dictionary containing event data
        prediction_result: Dictionary containing prediction results
    """
    global _recent_predictions
    try:
        # Add to in-memory cache
        _recent_predictions.append({
            'event_data': event_data,
            'prediction': prediction_result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Limit cache size
        
        if len(_recent_predictions) > 1000:
            _recent_predictions = _recent_predictions[-1000:]
        
        # Update metrics
        _performance_metrics['total_predictions'] += 1
        if prediction_result.get('is_anomaly', False):
            _performance_metrics['anomalies_detected'] += 1
        _performance_metrics['last_updated'] = datetime.utcnow().isoformat()
        
        # Log to database if available
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO prediction_logs 
                    (user_id, timestamp, action_type, is_anomaly, anomaly_score, prediction_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_data.get('user_id'),
                        event_data.get('timestamp'),
                        event_data.get('action_type'),
                        prediction_result.get('is_anomaly', False),
                        prediction_result.get('anomaly_score', 0.0),
                        prediction_result.get('prediction_time')
                    )
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not log prediction to database: {e}")
        
        # Log to file
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, 'predictions.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps({
                'user_id': event_data.get('user_id'),
                'timestamp': event_data.get('timestamp'),
                'action_type': event_data.get('action_type'),
                'is_anomaly': prediction_result.get('is_anomaly', False),
                'anomaly_score': prediction_result.get('anomaly_score', 0.0),
                'prediction_time': prediction_result.get('prediction_time')
            }) + '\n')
        
        logger.debug(f"Logged prediction for user {event_data.get('user_id')}")
        
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}")

def get_performance_metrics(time_window=None):
    """
    Get performance metrics for the model
    
    Args:
        time_window: Time window in hours (optional)
        
    Returns:
        Dictionary containing performance metrics
    """
    try:
        # If time window is specified, filter recent predictions
        if time_window:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window)
            filtered_predictions = [
                p for p in _recent_predictions 
                if datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) > cutoff_time
            ]
        else:
            filtered_predictions = _recent_predictions
        
        # Calculate metrics
        total = len(filtered_predictions)
        anomalies = sum(1 for p in filtered_predictions if p['prediction'].get('is_anomaly', False))
        
        # Calculate average anomaly score
        avg_score = np.mean([p['prediction'].get('anomaly_score', 0.0) for p in filtered_predictions]) if total > 0 else 0.0
        
        # Get metrics by action type
        action_metrics = {}
        for p in filtered_predictions:
            action_type = p['event_data'].get('action_type', 'unknown')
            if action_type not in action_metrics:
                action_metrics[action_type] = {
                    'total': 0,
                    'anomalies': 0
                }
            action_metrics[action_type]['total'] += 1
            if p['prediction'].get('is_anomaly', False):
                action_metrics[action_type]['anomalies'] += 1
        
        # Calculate percentages for action metrics
        for action_type in action_metrics:
            action_total = action_metrics[action_type]['total']
            action_metrics[action_type]['anomaly_rate'] = (
                action_metrics[action_type]['anomalies'] / action_total
                if action_total > 0 else 0.0
            )
        
        # Create metrics dictionary
        metrics = {
            'total_predictions': total,
            'anomalies_detected': anomalies,
            'anomaly_rate': anomalies / total if total > 0 else 0.0,
            'average_anomaly_score': float(avg_score),
            'action_metrics': action_metrics,
            'time_window_hours': time_window,
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return {
            'error': str(e),
            'total_predictions': _performance_metrics['total_predictions'],
            'anomalies_detected': _performance_metrics['anomalies_detected'],
            'last_updated': _performance_metrics['last_updated']
        }