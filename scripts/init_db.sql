-- Initialize ML Pipeline Database
-- This script sets up the initial database schema for the ML pipeline

-- Create extension for UUID generation if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create database schema for model predictions and feedback
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255) NOT NULL,
    input_text TEXT NOT NULL,
    prediction FLOAT NOT NULL,
    confidence FLOAT,
    processing_time_ms INTEGER,
    request_id UUID DEFAULT uuid_generate_v4()
);

-- Create table for model feedback
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    prediction_id INTEGER REFERENCES predictions(id),
    actual_sentiment FLOAT,
    user_feedback TEXT,
    feedback_type VARCHAR(50) DEFAULT 'user'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_model_name ON predictions(model_name);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction_id ON feedback(prediction_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);

-- Create table for monitoring data drift
CREATE TABLE IF NOT EXISTS drift_reports (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    report_type VARCHAR(50) NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE,
    drift_score FLOAT,
    report_data JSONB
);

CREATE INDEX IF NOT EXISTS idx_drift_reports_created_at ON drift_reports(created_at);
CREATE INDEX IF NOT EXISTS idx_drift_reports_type ON drift_reports(report_type);