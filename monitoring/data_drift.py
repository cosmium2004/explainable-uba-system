"""
Cloud UBA Phase 6 - Data Drift Detection
This file contains functions for detecting data drift.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import os
import json
from flask import current_app

# Get logger
logger = logging.getLogger(__name__)

# Reference distribution for features
_reference_distributions = {}

def initialize_reference_distributions(reference_data):
    """
    Initialize reference distributions for features
    
    Args:
        reference_data: DataFrame containing reference data
    """
    global _reference_distributions
    
    try:
        logger.info("Initializing reference distributions")
        
        for column in reference_data.columns:
            # Skip non-numeric columns
            if not np.issubdtype(reference_data[column].dtype, np.number):
                continue
                
            # Calculate distribution statistics
            _reference_distributions[column] = {
                'mean': float(reference_data[column].mean()),
                'std': float(reference_data[column].std()),
                'min': float(reference_data[column].min()),
                'max': float(reference_data[column].max()),
                'q1': float(reference_data[column].quantile(0.25)),
                'median': float(reference_data[column].quantile(0.5)),
                'q3': float(reference_data[column].quantile(0.75))
            }
        
        # Save reference distributions
        save_reference_distributions()
        
        logger.info(f"Reference distributions initialized for {len(_reference_distributions)} features")
        
    except Exception as e:
        logger.error(f"Error initializing reference distributions: {str(e)}")

def save_reference_distributions():
    """
    Save reference distributions to file
    """
    try:
        # Create directory if it doesn't exist
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save to file
        dist_file = os.path.join(model_dir, 'reference_distributions.json')
        with open(dist_file, 'w') as f:
            json.dump(_reference_distributions, f)
        
        logger.info(f"Reference distributions saved to {dist_file}")
        
    except Exception as e:
        logger.error(f"Error saving reference distributions: {str(e)}")

def load_reference_distributions():
    """
    Load reference distributions from file
    """
    global _reference_distributions
    
    try:
        # Load from file
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        dist_file = os.path.join(model_dir, 'reference_distributions.json')
        
        if os.path.exists(dist_file):
            with open(dist_file, 'r') as f:
                _reference_distributions = json.load(f)
            
            logger.info(f"Reference distributions loaded from {dist_file}")
            return True
        else:
            logger.warning(f"Reference distributions file not found: {dist_file}")
            return False
            
    except Exception as e:
        logger.error(f"Error loading reference distributions: {str(e)}")
        return False

def detect_data_drift(current_data):
    """
    Detect data drift in current data compared to reference data
    
    Args:
        current_data: DataFrame containing current data
        
    Returns:
        Dictionary containing drift detection results
    """
    try:
        # Load reference distributions if not already loaded
        if not _reference_distributions:
            if not load_reference_distributions():
                return {
                    'drift_detected': False,
                    'error': 'Reference distributions not available'
                }
        
        # Initialize results
        drift_results = {
            'drift_detected': False,
            'drifted_features': [],
            'drift_scores': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check each feature
        for column in current_data.columns:
            # Skip non-numeric columns or columns not in reference
            if column not in _reference_distributions or not np.issubdtype(current_data[column].dtype, np.number):
                continue
            
            # Get reference statistics
            ref_stats = _reference_distributions[column]
            
            # Calculate current statistics
            curr_mean = float(current_data[column].mean())
            curr_std = float(current_data[column].std())
            
            # Calculate drift score (normalized difference in means)
            if ref_stats['std'] > 0:
                drift_score = abs(curr_mean - ref_stats['mean']) / ref_stats['std']
            else:
                drift_score = abs(curr_mean - ref_stats['mean'])
            
            # Store drift score
            drift_results['drift_scores'][column] = float(drift_score)
            
            # Check if drift is significant
            if drift_score > 0.5:  # Threshold can be adjusted
                drift_results['drifted_features'].append({
                    'feature': column,
                    'drift_score': float(drift_score),
                    'reference_mean': ref_stats['mean'],
                    'current_mean': curr_mean,
                    'reference_std': ref_stats['std'],
                    'current_std': curr_std
                })
        
        # Set drift detected flag
        if drift_results['drifted_features']:
            drift_results['drift_detected'] = True
            logger.warning(f"Data drift detected in {len(drift_results['drifted_features'])} features")
        
        return drift_results
        
    except Exception as e:
        logger.error(f"Error detecting data drift: {str(e)}")
        return {
            'drift_detected': False,
            'error': str(e)
        }