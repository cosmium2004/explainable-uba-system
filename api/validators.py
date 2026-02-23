"""
Cloud UBA Phase 6 - API Validators
This file contains validation functions for API inputs.
"""

import logging
from datetime import datetime
import re
from jsonschema import validate, ValidationError

# Get logger
logger = logging.getLogger(__name__)

# Schema for prediction input
prediction_schema = {
    "type": "object",
    "required": ["user_id", "timestamp", "action_type"],
    "properties": {
        "user_id": {"type": "string", "minLength": 1},
        "timestamp": {"type": "string", "format": "date-time"},
        "action_type": {"type": "string", "enum": ["login", "logout", "download", "upload", "delete", "view", "admin"]},
        "resource_id": {"type": "string"},
        "ip_address": {"type": "string"},
        "data_volume_mb": {"type": "number", "minimum": 0},
        "location": {"type": "string"},
        "device_type": {"type": "string"},
        "failed_attempts": {"type": "integer", "minimum": 0},
        "location_change": {"type": "boolean"},
        "explain": {"type": "boolean"}
    },
    "additionalProperties": True
}

# Schema for batch input
batch_schema = {
    "type": "object",
    "required": ["instances"],
    "properties": {
        "instances": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1000,
            "items": prediction_schema
        },
        "explain": {"type": "boolean"}
    },
    "additionalProperties": False
}

# Schema for model update
model_update_schema = {
    "type": "object",
    "required": ["version", "model_path"],
    "properties": {
        "version": {"type": "string", "minLength": 1},
        "model_path": {"type": "string", "minLength": 1},
        "is_active": {"type": "boolean"},
        "deployed_by": {"type": "string"},
        "performance_metrics": {"type": "object"}
    },
    "additionalProperties": False
}

def validate_prediction_input(data):
    """
    Validate prediction input data
    
    Args:
        data: Dictionary containing input data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate against schema
        validate(instance=data, schema=prediction_schema)
        
        # Additional validation
        if 'timestamp' in data:
            try:
                # Check if timestamp is valid
                datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                return False, "Invalid timestamp format"
        
        if 'ip_address' in data:
            # Simple IP address validation
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            if not re.match(ip_pattern, data['ip_address']):
                return False, "Invalid IP address format"
        
        return True, None
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Error validating prediction input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_batch_input(data):
    """
    Validate batch input data
    
    Args:
        data: Dictionary containing input data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate against schema
        validate(instance=data, schema=batch_schema)
        
        # Validate each instance
        for i, instance in enumerate(data['instances']):
            is_valid, error = validate_prediction_input(instance)
            if not is_valid:
                return False, f"Invalid instance at index {i}: {error}"
        
        return True, None
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Error validating batch input: {str(e)}")
        return False, f"Validation error: {str(e)}"

def validate_model_update(data):
    """
    Validate model update data
    
    Args:
        data: Dictionary containing input data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Validate against schema
        validate(instance=data, schema=model_update_schema)
        
        # Additional validation
        import os
        if not os.path.exists(data['model_path']):
            return False, f"Model file not found: {data['model_path']}"
        
        return True, None
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Error validating model update: {str(e)}")
        return False, f"Validation error: {str(e)}"