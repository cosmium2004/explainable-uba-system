"""
Cloud UBA Phase 6 - Database Queries
This file contains common database queries.
"""

import logging
from datetime import datetime, timedelta
import json
from database.connector import get_db_connection

# Get logger
logger = logging.getLogger(__name__)

def get_user(user_id):
    """
    Get user by ID
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary containing user data or None if not found
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )
        
        user = cursor.fetchone()
        
        if user:
            return dict(user)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}")
        return None

def get_user_events(user_id, limit=100, offset=0):
    """
    Get events for a user
    
    Args:
        user_id: User ID
        limit: Maximum number of events to return
        offset: Offset for pagination
        
    Returns:
        List of dictionaries containing event data
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT * FROM events 
            WHERE user_id = ? 
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, limit, offset)
        )
        
        events = cursor.fetchall()
        
        return [dict(event) for event in events]
        
    except Exception as e:
        logger.error(f"Error getting events for user {user_id}: {str(e)}")
        return []

def get_user_alerts(user_id, resolved=None, limit=100, offset=0):
    """
    Get alerts for a user
    
    Args:
        user_id: User ID
        resolved: Filter by resolved status (None for all)
        limit: Maximum number of alerts to return
        offset: Offset for pagination
        
    Returns:
        List of dictionaries containing alert data
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM alerts 
            WHERE user_id = ?
        """
        
        params = [user_id]
        
        if resolved is not None:
            query += " AND is_resolved = ?"
            params.append(1 if resolved else 0)
        
        query += " ORDER BY alert_time DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        alerts = cursor.fetchall()
        
        return [dict(alert) for alert in alerts]
        
    except Exception as e:
        logger.error(f"Error getting alerts for user {user_id}: {str(e)}")
        return []

def save_event(event_data):
    """
    Save an event to the database
    
    Args:
        event_data: Dictionary containing event data
        
    Returns:
        Boolean indicating success
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO events 
            (user_id, timestamp, action_type, resource_id, ip_address, data_volume_mb, location, device_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_data.get('user_id'),
                event_data.get('timestamp', datetime.utcnow().isoformat()),
                event_data.get('action_type'),
                event_data.get('resource_id'),
                event_data.get('ip_address'),
                event_data.get('data_volume_mb', 0.0),
                event_data.get('location'),
                event_data.get('device_type')
            )
        )
        
        conn.commit()
        logger.debug(f"Event saved for user {event_data.get('user_id')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving event: {str(e)}")
        return False

def save_alert(alert_data):
    """
    Save an alert to the database
    
    Args:
        alert_data: Dictionary containing alert data
        
    Returns:
        Boolean indicating success
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT INTO alerts 
            (user_id, timestamp, action_type, resource_id, anomaly_score, alert_time, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_data.get('user_id'),
                alert_data.get('timestamp'),
                alert_data.get('action_type'),
                alert_data.get('resource_id'),
                alert_data.get('anomaly_score'),
                alert_data.get('alert_time', datetime.utcnow().isoformat()),
                alert_data.get('severity', 'medium')
            )
        )
        
        conn.commit()
        logger.info(f"Alert saved for user {alert_data.get('user_id')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving alert: {str(e)}")
        return False

def resolve_alert(alert_id, resolution_data):
    """
    Mark an alert as resolved
    
    Args:
        alert_id: Alert ID
        resolution_data: Dictionary containing resolution data
        
    Returns:
        Boolean indicating success
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            UPDATE alerts
            SET is_resolved = 1,
                resolution_notes = ?,
                resolved_by = ?,
                resolved_at = ?
            WHERE alert_id = ?
            """,
            (
                resolution_data.get('notes'),
                resolution_data.get('resolved_by'),
                resolution_data.get('resolved_at', datetime.utcnow().isoformat()),
                alert_id
            )
        )
        
        conn.commit()
        
        if cursor.rowcount > 0:
            logger.info(f"Alert {alert_id} marked as resolved")
            return True
        else:
            logger.warning(f"Alert {alert_id} not found")
            return False
        
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {str(e)}")
        return False

def get_model_version(version=None):
    """
    Get model version information
    
    Args:
        version: Specific version to get (None for active)
        
    Returns:
        Dictionary containing model version data or None if not found
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if version:
            cursor.execute(
                "SELECT * FROM model_versions WHERE version = ?",
                (version,)
            )
        else:
            cursor.execute(
                "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY deployed_at DESC LIMIT 1"
            )
        
        model_version = cursor.fetchone()
        
        if model_version:
            return dict(model_version)
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error getting model version: {str(e)}")
        return None

def save_model_version(version_data):
    """
    Save a new model version
    
    Args:
        version_data: Dictionary containing model version data
        
    Returns:
        Boolean indicating success
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Deactivate current active model if setting this one as active
        if version_data.get('is_active', False):
            cursor.execute(
                "UPDATE model_versions SET is_active = 0 WHERE is_active = 1"
            )
        
        # Insert new model version
        cursor.execute(
            """
            INSERT INTO model_versions 
            (version, model_path, deployed_at, is_active, performance_metrics, deployed_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                version_data.get('version'),
                version_data.get('model_path'),
                version_data.get('deployed_at', datetime.utcnow().isoformat()),
                1 if version_data.get('is_active', False) else 0,
                json.dumps(version_data.get('performance_metrics', {})),
                version_data.get('deployed_by')
            )
        )
        
        conn.commit()
        logger.info(f"Model version {version_data.get('version')} saved")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving model version: {str(e)}")
        return False

def get_statistics(days=30):
    """
    Get system statistics
    
    Args:
        days: Number of days to include in statistics
        
    Returns:
        Dictionary containing statistics
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate date cutoff
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get event counts
        cursor.execute(
            """
            SELECT action_type, COUNT(*) as count
            FROM events
            WHERE timestamp >= ?
            GROUP BY action_type
            """,
            (cutoff_date,)
        )
        
        event_counts = {row['action_type']: row['count'] for row in cursor.fetchall()}
        
        # Get alert counts by severity
        cursor.execute(
            """
            SELECT severity, COUNT(*) as count
            FROM alerts
            WHERE alert_time >= ?
            GROUP BY severity
            """,
            (cutoff_date,)
        )
        
        alert_counts = {row['severity']: row['count'] for row in cursor.fetchall()}
        
        # Get user counts
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE is_active = 1")
        active_users = cursor.fetchone()['count']
        
        # Get total events
        cursor.execute(
            "SELECT COUNT(*) as count FROM events WHERE timestamp >= ?",
            (cutoff_date,)
        )
        total_events = cursor.fetchone()['count']
        
        # Get total alerts
        cursor.execute(
            "SELECT COUNT(*) as count FROM alerts WHERE alert_time >= ?",
            (cutoff_date,)
        )
        total_alerts = cursor.fetchone()['count']
        
        # Get resolved alerts
        cursor.execute(
            """
            SELECT COUNT(*) as count 
            FROM alerts 
            WHERE alert_time >= ? AND is_resolved = 1
            """,
            (cutoff_date,)
        )
        resolved_alerts = cursor.fetchone()['count']
        
        # Create statistics
        statistics = {
            'time_period_days': days,
            'active_users': active_users,
            'total_events': total_events,
            'total_alerts': total_alerts,
            'resolved_alerts': resolved_alerts,
            'resolution_rate': resolved_alerts / total_alerts if total_alerts > 0 else 0,
            'event_counts': event_counts,
            'alert_counts': alert_counts,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return {
            'error': str(e),
            'time_period_days': days,
            'generated_at': datetime.utcnow().isoformat()
        }