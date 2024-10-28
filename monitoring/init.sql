-- monitoring/init.sql

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    user_query TEXT NOT NULL,
    thumbs_up BOOLEAN,
    thumbs_down BOOLEAN,
    relevant BOOLEAN,
    model_used TEXT,
    response_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
