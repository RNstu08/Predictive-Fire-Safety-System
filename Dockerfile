# Dockerfile for HVS Prediction API

# Use an official Python runtime as a parent image
# Using a specific version is recommended for reproducibility
# Choose a slim version to keep the image size smaller
FROM python:3.11-slim

# Set environment variables
# Prevents Python from writing pyc files to disc (optional)
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain Python libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*
# (No specific system dependencies needed for our current setup)

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
# This includes the 'src' directory where our api, services, etc., live
COPY ./src ./src
# Copy other necessary files if any (e.g., config.yaml if used by the app)
# COPY config.yaml .

# Make port 5001 available to the world outside this container
# This is the port our Flask app runs on (defined in app.py)
EXPOSE 5001

# Define environment variable for the MLflow Run ID (best practice)
# The actual value will be passed during 'docker run' or via Kubernetes config
ENV BEST_MLFLOW_RUN_ID="RUN_ID_NOT_SET"


# Tell MLflow where to find the mlruns directory inside the container
ENV MLFLOW_TRACKING_URI="file:///app/mlruns"
    
# Define the command to run your application
# This runs the Flask development server directly via python
# For production, you'd typically use a WSGI server like Gunicorn:
# CMD ["gunicorn", "--bind", "0.0.0.0:5001", "src.api.app:create_app()"]
# Using python directly for simplicity in this phase:
# CMD ["python", "src/api/app.py"]
CMD ["python", "-m", "src.api.app"]

