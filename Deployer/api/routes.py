"""
Cloud UBA Phase 6 - API Routes
This file contains the API routes for the Cloud UBA application.
"""

from flask import Blueprint, request, jsonify, current_app
import logging
from models.prediction import predict_anomaly, batch_predict
from models.model_loader import get_model, reload_model
from explainers.report_generator import generate_explanation
from monitoring.performance_tracker import log_prediction, get_performance_metrics
from monitoring.data_drift import detect_data_drift
from monitoring.alerting import generate_alert
from api.middleware import token_required, validate_input
from api.validators import validate_prediction_input, validate_batch_input, validate_model_update
from database.queries import get_statistics, save_model_version
import pandas as pd

# Create blueprint
api_blueprint = Blueprint('api', __name__)

# Get logger
logger = logging.getLogger(__name__)

@api_blueprint.route('/predict', methods=['POST'])
@token_required
@validate_input(validate_prediction_input)
def predict():
    """
    Endpoint for real-time anomaly detection
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Make prediction
        result = predict_anomaly(data)
        
        # Log prediction for monitoring
        log_prediction(data, result)
        
        # Check if explanation is requested
        if data.get('explain', False):
            explanation = generate_explanation(data, result)
            result['explanation'] = explanation
        
        # Generate alert if needed
        if result.get('is_anomaly', False) and result.get('anomaly_score', 0) >= current_app.config.get('ALERT_THRESHOLD', 0.8):
            explanation = explanation if data.get('explain', False) else generate_explanation(data, result)
            generate_alert(data, result, explanation)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@api_blueprint.route('/batch', methods=['POST'])
@token_required
@validate_input(validate_batch_input)
def batch():
    """
    Endpoint for batch processing of historical data
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Process batch
        results = batch_predict(data['instances'])
        
        # Log predictions and generate explanations if requested
        for i, result in enumerate(results):
            event_data = data['instances'][i]
            log_prediction(event_data, result)
            
            if data.get('explain', False):
                explanation = generate_explanation(event_data, result)
                result['explanation'] = explanation
            
            # Generate alert if needed
            if result.get('is_anomaly', False) and result.get('anomaly_score', 0) >= current_app.config.get('ALERT_THRESHOLD', 0.8):
                explanation = explanation if data.get('explain', False) else generate_explanation(event_data, result)
                generate_alert(event_data, result, explanation)
        
        # Check for data drift
        if len(data['instances']) >= 10:
            # Extract features for drift detection
            features_list = []
            for event in data['instances']:
                from utils.preprocessing import extract_features_from_event
                features = extract_features_from_event(event)
                features_list.append(features)
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            
            # Detect drift
            drift_results = detect_data_drift(features_df)
            
            # Include drift results in response if drift detected
            if drift_results.get('drift_detected', False):
                logger.warning("Data drift detected in batch processing")
        
        return jsonify({
            'success': True,
            'predictions': results,
            'count': len(results),
            'anomalies': sum(1 for r in results if r.get('is_anomaly', False))
        })
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Batch processing failed',
            'message': str(e)
        }), 500


@api_blueprint.route('/model', methods=['GET'])
@token_required
def get_model_info():
    """
    Endpoint for getting model information
    """
    try:
        # Get model
        model = get_model()
        
        # Get model type
        model_type = type(model).__name__
        
        # Get feature names if available
        feature_names = []
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        
        # Get model parameters
        params = model.get_params()
        
        # Get model version from database
        from database.queries import get_model_version
        model_version = get_model_version()
        
        # Create response
        response = {
            'success': True,
            'model_info': {
                'model_type': model_type,
                'feature_count': len(feature_names),
                'parameters': params
            }
        }
        
        # Add version information if available
        if model_version:
            response['model_info']['version'] = model_version['version']
            response['model_info']['deployed_at'] = model_version['deployed_at']
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error getting model information: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get model information',
            'message': str(e)
        }), 500


@api_blueprint.route('/model', methods=['POST'])
@token_required
@validate_input(validate_model_update)
def update_model():
    """
    Endpoint for updating the model
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Save model version to database
        success = save_model_version(data)
        
        if not success:
            return jsonify({
                'success': False,
                'error': 'Failed to save model version'
            }), 500
        
        # Reload model
        reload_model()
        
        return jsonify({
            'success': True,
            'message': f"Model updated to version {data['version']}"
        })
    
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Model update failed',
            'message': str(e)
        }), 500


@api_blueprint.route('/metrics', methods=['GET'])
@token_required
def get_metrics():
    """
    Endpoint for getting performance metrics
    """
    try:
        # Get time window from query parameters
        time_window = request.args.get('time_window')
        if time_window:
            time_window = int(time_window)
        
        # Get performance metrics
        metrics = get_performance_metrics(time_window)
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get performance metrics',
            'message': str(e)
        }), 500


@api_blueprint.route('/statistics', methods=['GET'])
@token_required
def get_system_statistics():
    """
    Endpoint for getting system statistics
    """
    try:
        # Get days from query parameters
        days = request.args.get('days', 30)
        days = int(days)
        
        # Get statistics
        statistics = get_statistics(days)
        
        return jsonify({
            'success': True,
            'statistics': statistics
        })
    
    except Exception as e:
        logger.error(f"Error getting system statistics: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get system statistics',
            'message': str(e)
        }), 500


@api_blueprint.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint that doesn't require authentication
    """
    return jsonify({
        'success': True,
        'status': 'healthy',
        'version': current_app.config['VERSION']
    })