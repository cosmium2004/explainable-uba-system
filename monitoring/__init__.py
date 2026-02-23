"""
Cloud UBA Phase 6 - Monitoring Package
This package contains monitoring functionality for the Cloud UBA application.
"""

from monitoring.logger import setup_logging
from monitoring.alerting import generate_alert, get_severity_level
from monitoring.data_drift import detect_data_drift, initialize_reference_distributions
from monitoring.performance_tracker import log_prediction, get_performance_metrics