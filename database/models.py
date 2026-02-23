"""
Cloud UBA Phase 6 - Database Models
This file contains database models/schema definitions.
"""

import sqlite3
import logging
from datetime import datetime

# Get logger
logger = logging.getLogger(__name__)

def create_tables(conn):
    """
    Create database tables if they don't exist
    
    Args:
        conn: Database connection
    """
    try:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            email TEXT,
            department TEXT,
            role TEXT,
            created_at TEXT NOT NULL,
            last_login TEXT,
            is_active INTEGER DEFAULT 1
        )
        ''')
        
        # Create events table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            action_type TEXT NOT NULL,
            resource_id TEXT,
            ip_address TEXT,
            data_volume_mb REAL,
            location TEXT,
            device_type TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Create prediction_logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            action_type TEXT NOT NULL,
            is_anomaly INTEGER NOT NULL,
            anomaly_score REAL NOT NULL,
            prediction_time TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Create alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            action_type TEXT NOT NULL,
            resource_id TEXT,
            anomaly_score REAL NOT NULL,
            alert_time TEXT NOT NULL,
            severity TEXT NOT NULL,
            is_resolved INTEGER DEFAULT 0,
            resolution_notes TEXT,
            resolved_by TEXT,
            resolved_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
        ''')
        
        # Create model_versions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id INTEGER PRIMARY KEY AUTOINCREMENT,
            version TEXT NOT NULL,
            model_path TEXT NOT NULL,
            deployed_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            performance_metrics TEXT,
            deployed_by TEXT
        )
        ''')
        
        conn.commit()
        logger.info("Database tables created successfully")
        
    except sqlite3.Error as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

def insert_sample_data(conn):
    """
    Insert sample data for testing
    
    Args:
        conn: Database connection
    """
    try:
        cursor = conn.cursor()
        
        # Check if users table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        if user_count == 0:
            # Insert sample users
            users = [
                ('user-001', 'john.doe', 'john.doe@example.com', 'IT', 'admin', datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), 1),
                ('user-002', 'jane.smith', 'jane.smith@example.com', 'Finance', 'analyst', datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), 1),
                ('user-003', 'bob.johnson', 'bob.johnson@example.com', 'HR', 'manager', datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), 1)
            ]
            
            cursor.executemany('''
            INSERT INTO users (user_id, username, email, department, role, created_at, last_login, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', users)
            
            # Insert sample events
            events = [
                ('user-001', datetime.utcnow().isoformat(), 'login', None, '192.168.1.1', 0.0, 'New York', 'desktop'),
                ('user-001', datetime.utcnow().isoformat(), 'download', 'file-001', '192.168.1.1', 2.5, 'New York', 'desktop'),
                ('user-002', datetime.utcnow().isoformat(), 'login', None, '10.0.0.1', 0.0, 'Chicago', 'mobile'),
                ('user-003', datetime.utcnow().isoformat(), 'admin', 'sys-001', '172.16.0.1', 0.0, 'Dallas', 'desktop')
            ]
            
            cursor.executemany('''
            INSERT INTO events (user_id, timestamp, action_type, resource_id, ip_address, data_volume_mb, location, device_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', events)
            
            conn.commit()
            logger.info("Sample data inserted successfully")
        else:
            logger.info("Database already contains data, skipping sample data insertion")
        
    except sqlite3.Error as e:
        logger.error(f"Error inserting sample data: {str(e)}")
        raise