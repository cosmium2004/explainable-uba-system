"""
Cloud UBA Phase 6 - Database Package
This package contains database functionality for the Cloud UBA application.
"""

from database.connector import get_db_connection, close_db_connection, init_app
from database.models import create_tables, insert_sample_data