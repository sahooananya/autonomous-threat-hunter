# AI-Powered Cybersecurity Threat Detector
AI-powered agent for autonomous cybersecurity threat detection using ML and LLMs.
This project implements a full-stack cybersecurity threat detector using Hugging Face Transformers to analyze network traffic and system logs. It classifies logs as either "malicious" or "normal".

## Features

*   **Model Training**: Fine-tunes a DistilBERT model for binary classification on network log data.
*   **Inference API**: FastAPI backend provides a `/predict` endpoint to classify log entries.
*   **User Interface**: React frontend allows users to input log data (as JSON or structured text) and see predictions.
*   **Dockerized**: Fully containerized using Docker and Docker Compose for easy deployment.
*   **(Optional) Kafka Integration**: Basic setup for future integration with Apache Kafka for real-time log stream processing.

## Tech Stack

*   **Frontend**: React.js
*   **Backend**: FastAPI (Python)
*   **ML**: Hugging Face Transformers (DistilBERT), PyTorch, Scikit-learn
*   **Deployment**: Docker, Docker Compose
*   **(Optional)**: Apache Kafka

## Project Structure
/
├── ml_model/
│ ├── train_model.py # Script to train and save the model
│ ├── model_utils.py # Utilities for loading model and prediction
│ └── saved_transformer/ # Directory for saved model and tokenizer
├── backend/
│ ├── main.py # FastAPI application
│ └── requirements.txt # Python dependencies for backend
├── frontend/
│ ├── src/ # React application source
│ ├── public/
│ ├── package.json
│ └── Dockerfile # Dockerfile for frontend
├── docker/
│ ├── kafka/ # (Optional) Kafka configurations
│ └── docker-compose.yml # Docker Compose configuration
└── README.md
