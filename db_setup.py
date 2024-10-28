import psycopg2
import os

def init_db():
    # Connect to PostgreSQL database
    db_connection = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )
    cursor = db_connection.cursor()

    # Create the feedback table if it does not exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        user_query TEXT,
        response TEXT,
        model_used VARCHAR(50),
        relevance BOOLEAN,
        thumbs_up INT DEFAULT 0,
        thumbs_down INT DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    db_connection.commit()
    cursor.close()
    db_connection.close()

# Call this function when the script runs
if __name__ == "__main__":
    init_db()
