"""
Cloud UBA Phase 6 - Alerting System
This file contains functions for generating alerts for critical anomalies.
"""

import logging
import json
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import current_app

# Get logger
logger = logging.getLogger(__name__)

def generate_alert(event_data, prediction_result, explanation=None):
    """
    Generate an alert for a critical anomaly
    
    Args:
        event_data: Dictionary containing event data
        prediction_result: Dictionary containing prediction results
        explanation: Dictionary containing explanation (optional)
        
    Returns:
        Boolean indicating if alert was generated
    """
    try:
        # Check if alert should be generated
        threshold = current_app.config.get('ALERT_THRESHOLD', 0.8)
        
        if not prediction_result.get('is_anomaly', False):
            return False
            
        anomaly_score = prediction_result.get('anomaly_score', 0.0)
        if anomaly_score < threshold:
            return False
        
        logger.info(f"Generating alert for user {event_data.get('user_id')} with score {anomaly_score}")
        
        # Create alert data
        alert = {
            'user_id': event_data.get('user_id'),
            'timestamp': event_data.get('timestamp'),
            'action_type': event_data.get('action_type'),
            'resource_id': event_data.get('resource_id'),
            'anomaly_score': anomaly_score,
            'alert_time': datetime.utcnow().isoformat() + 'Z',
            'severity': get_severity_level(anomaly_score)
        }
        
        # Add explanation if available
        if explanation:
            alert['explanation'] = explanation.get('text_explanation', '')
        
        # Log alert
        log_alert(alert)
        
        # Send alert notification
        send_alert_notification(alert)
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating alert: {str(e)}")
        return False

def get_severity_level(anomaly_score):
    """
    Get severity level based on anomaly score
    
    Args:
        anomaly_score: Anomaly score
        
    Returns:
        String representing severity level
    """
    if anomaly_score >= 0.9:
        return 'critical'
    elif anomaly_score >= 0.8:
        return 'high'
    elif anomaly_score >= 0.7:
        return 'medium'
    else:
        return 'low'

def log_alert(alert):
    """
    Log alert to file
    
    Args:
        alert: Dictionary containing alert data
    """
    try:
        # Create directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Log to file
        alert_file = os.path.join(log_dir, 'alerts.jsonl')
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
        
        logger.info(f"Alert logged to {alert_file}")
        
    except Exception as e:
        logger.error(f"Error logging alert: {str(e)}")

def send_alert_notification(alert):
    """
    Send alert notification
    
    Args:
        alert: Dictionary containing alert data
    """
    try:
        # Check if email notifications are enabled
        email_enabled = current_app.config.get('EMAIL_NOTIFICATIONS_ENABLED', False)
        if not email_enabled:
            logger.info("Email notifications are disabled")
            return
        
        # Get email configuration
        smtp_server = current_app.config.get('SMTP_SERVER')
        smtp_port = current_app.config.get('SMTP_PORT')
        smtp_username = current_app.config.get('SMTP_USERNAME')
        smtp_password = current_app.config.get('SMTP_PASSWORD')
        sender_email = current_app.config.get('SENDER_EMAIL')
        recipient_emails = current_app.config.get('RECIPIENT_EMAILS', [])
        
        if not smtp_server or not sender_email or not recipient_emails:
            logger.warning("Email configuration incomplete")
            return
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipient_emails)
        msg['Subject'] = f"Cloud UBA Alert: {alert['severity'].upper()} severity anomaly detected"
        
        # Email body
        body = f"""
        <html>
        <body>
            <h2>Cloud UBA Security Alert</h2>
            <p><strong>Severity:</strong> {alert['severity'].upper()}</p>
            <p><strong>User ID:</strong> {alert['user_id']}</p>
            <p><strong>Action:</strong> {alert['action_type']}</p>
            <p><strong>Resource:</strong> {alert['resource_id']}</p>
            <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
            <p><strong>Anomaly Score:</strong> {alert['anomaly_score']:.4f}</p>
            
            <h3>Explanation</h3>
            <p>{alert.get('explanation', 'No explanation available')}</p>
            
            <p>Please investigate this activity immediately.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            if smtp_username and smtp_password:
                server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Alert notification sent to {', '.join(recipient_emails)}")
        
    except Exception as e:
        logger.error(f"Error sending alert notification: {str(e)}")