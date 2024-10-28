# monitoring/init_db.py

import os
import psycopg2
from time import sleep

# Define connection parameters
db_params = {
    'dbname': os.getenv('POSTGRES_DB', 'monitoring_db'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'example'),
    'host': os.getenv('POSTGRES_HOST', 'postgres'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

# Wait for the PostgreSQL server to be ready
def wait_for_postgres():
    while True:
        try:
            conn = psycopg2.connect(**db_params)
            conn.close()
            break
        except psycopg2.OperationalError:
            print("Waiting for PostgreSQL to be ready...")
            sleep(5)

# Create the feedback table
def create_table():
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        user_query TEXT NOT NULL,
        thumbs_up BOOLEAN,
        thumbs_down BOOLEAN,
        relevant BOOLEAN,
        model_used TEXT,
        response_time FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Table created successfully.")

if __name__ == '__main__':
    wait_for_postgres()
    create_table()
