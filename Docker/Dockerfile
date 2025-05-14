# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./backend/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code into the container at /app/backend
COPY ./backend /app/backend

# Copy the ml_model directory into the container at /app/ml_model
# This assumes the model is already trained and present in ./ml_model/saved_transformer
# If you want to train inside Docker, you'd also copy train_model.py and data
COPY ./ml_model /app/ml_model

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (can be overridden by docker-compose)
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/ml_model/saved_transformer/

# Command to run the application using Uvicorn
# The entrypoint will be backend.main:app as main.py is inside backend/
# CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
# The command is often specified in docker-compose.yml to allow for easier changes (e.g., --reload for dev)