"""
Cloud UBA Phase 6 - Preprocessing Utilities
This file contains utilities for preprocessing event data.
"""

import numpy as np
from datetime import datetime
import logging
import re

# Get logger
logger = logging.getLogger(__name__)

def extract_features_from_event(event):
    """
    Extract features from an event for prediction
    
    Args:
        event: Dictionary containing event data
        
    Returns:
        Dictionary of features
    """
    features = {}
    
    try:
        # Basic event information
        features['user_id'] = event.get('user_id', '')
        
        # Process timestamp
        if 'timestamp' in event:
            dt = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            features['hour_of_day'] = dt.hour
            features['day_of_week'] = dt.weekday()
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            
            # Time of day categories
            if 5 <= dt.hour < 12:
                features['time_of_day'] = 'morning'
            elif 12 <= dt.hour < 17:
                features['time_of_day'] = 'afternoon'
            elif 17 <= dt.hour < 22:
                features['time_of_day'] = 'evening'
            else:
                features['time_of_day'] = 'night'
        else:
            # Default values if timestamp is missing
            features['hour_of_day'] = 12
            features['day_of_week'] = 0
            features['is_weekend'] = 0
            features['time_of_day'] = 'afternoon'
        
        # Action type one-hot encoding
        action_type = event.get('action_type', 'unknown')
        for action in ['login', 'logout', 'download', 'upload', 'delete', 'view', 'admin']:
            features[f'action_type_{action}'] = 1 if action_type == action else 0
        
        # Data volume
        features['data_volume_mb'] = float(event.get('data_volume_mb', 0))
        
        # Location change
        features['location_change'] = 1 if event.get('location_change', False) else 0
        
        # Failed attempts
        features['failed_attempts'] = int(event.get('failed_attempts', 0))
        
        # Device type
        device_type = event.get('device_type', 'unknown')
        for device in ['desktop', 'mobile', 'tablet', 'server']:
            features[f'device_type_{device}'] = 1 if device_type == device else 0
        
        # IP address features
        ip = event.get('ip_address', '')
        features['is_internal_ip'] = 1 if is_internal_ip(ip) else 0
        
        # Resource type
        resource_type = get_resource_type(event.get('resource_id', ''))
        for res_type in ['file', 'database', 'application', 'system']:
            features[f'resource_type_{res_type}'] = 1 if resource_type == res_type else 0
        
        # Additional features can be added here
        
        logger.debug(f"Extracted {len(features)} features from event")
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        # Return basic features to avoid complete failure
        return {
            'user_id': event.get('user_id', ''),
            'hour_of_day': 12,
            'day_of_week': 0,
            'is_weekend': 0,
            'action_type_login': 0,
            'data_volume_mb': 0,
            'failed_attempts': 0
        }
    
    return features

def is_internal_ip(ip):
    """
    Check if an IP address is internal
    
    Args:
        ip: IP address string
        
    Returns:
        Boolean indicating if the IP is internal
    """
    if not ip:
        return False
        
    # Check for private IP ranges
    private_patterns = [
        r'^10\.',
        r'^172\.(1[6-9]|2[0-9]|3[0-1])\.',
        r'^192\.168\.'
    ]
    
    for pattern in private_patterns:
        if re.match(pattern, ip):
            return True
            
    return False

def get_resource_type(resource_id):
    """
    Determine resource type from resource ID
    
    Args:
        resource_id: Resource ID string
        
    Returns:
        String representing the resource type
    """
    if not resource_id:
        return 'unknown'
        
    # Simple heuristic based on resource ID prefix
    if resource_id.startswith('file-'):
        return 'file'
    elif resource_id.startswith('db-'):
        return 'database'
    elif resource_id.startswith('app-'):
        return 'application'
    elif resource_id.startswith('sys-'):
        return 'system'
    else:
        return 'unknown'