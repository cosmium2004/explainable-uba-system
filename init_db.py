from app import create_app
from database.connector import get_db_connection

if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        conn = get_db_connection()
        print("Database initialized successfully!")