from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field  # Pydantic v2
from typing import Union, Dict, Any
import logging
import sys
import os

# Add ml_model directory to Python path to import model_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ml_model.model_utils import predict_threat, load_model  # Eagerly load model on startup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Cybersecurity Threat Detector API")

# CORS (Cross-Origin Resource Sharing)
# Allows requests from your React frontend (running on a different port)
origins = [
    "http://localhost",
    "http://localhost:3000",  # Default React dev server
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Pydantic Models for Request/Response ---
class LogEntryText(BaseModel):
    log_entry: str = Field(..., example="srcip:192.168.1.10 dstip:10.0.0.5 sport:54321 dport:80 proto:tcp state:FIN")


class LogEntryDict(BaseModel):
    log_data: Dict[str, Any] = Field(..., example={
        "srcip": "192.168.1.10", "dstip": "10.0.0.5", "sport": 54321, "dport": 80, "proto": "tcp"
    })


class PredictionResponse(BaseModel):
    prediction: str  # "malicious" or "normal" or "Error: ..."


# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Loading ML model...")
    try:
        load_model()  # Pre-load the model
        logging.info("ML model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load ML model on startup: {e}")
        # Depending on policy, you might want the app to not start,
        # or to operate in a degraded mode. For now, it will log and continue.


@app.get("/")
async def root():
    return {"message": "Cybersecurity Threat Detector API is running."}


@app.post("/predict", response_model=PredictionResponse)
async def predict_log_threat(log_input: Union[LogEntryText, LogEntryDict]):
    """
    Accepts a log entry (either as a single string or a structured dictionary)
    and returns a prediction.
    """
    logging.info(f"Received prediction request: {log_input}")
    try:
        if isinstance(log_input, LogEntryText):
            data_to_predict = log_input.log_entry
        elif isinstance(log_input, LogEntryDict):
            data_to_predict = log_input.log_data
        else:
            # This case should ideally not be reached due to Pydantic validation of Union
            raise HTTPException(status_code=400, detail="Invalid input type. Must be LogEntryText or LogEntryDict.")

        prediction_result = predict_threat(data_to_predict)

        if "Error:" in prediction_result:  # Check if predict_threat returned an error string
            # Log the error server-side as well
            logging.error(f"Prediction error for input {data_to_predict}: {prediction_result}")
            # Return a more generic error to the client or the specific one
            raise HTTPException(status_code=500, detail=prediction_result)

        return PredictionResponse(prediction=prediction_result)

    except FileNotFoundError as e:  # Specifically for model not found
        logging.error(f"Model file not found during prediction: {e}")
        raise HTTPException(status_code=503,
                            detail=f"Model not available: {str(e)}. Please ensure the model is trained and accessible.")
    except HTTPException as e:  # Re-raise HTTPExceptions
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# --- (Optional) Kafka Consumer ---
# Needs kafka-python: pip install kafka-python
# from kafka import KafkaConsumer
# import json
# KAFKA_TOPIC = "network_logs"
# KAFKA_BROKERS = "kafka:9092" # Assuming Kafka runs in Docker service named 'kafka'

# async def consume_kafka_logs():
#     # This would run in a background task or a separate process
#     logging.info(f"Attempting to connect to Kafka: {KAFKA_BROKERS} on topic {KAFKA_TOPIC}")
#     try:
#         consumer = KafkaConsumer(
#             KAFKA_TOPIC,
#             bootstrap_servers=KAFKA_BROKERS,
#             auto_offset_reset='earliest', # Start reading at the earliest message if new consumer group
#             group_id='threat-detector-group', # Consumer group ID
#             value_deserializer=lambda x: json.loads(x.decode('utf-8')) # Assuming JSON messages
#         )
#         logging.info("Successfully connected to Kafka. Waiting for messages...")
#         for message in consumer:
#             log_data = message.value # This should be a dict or string
#             logging.info(f"Received from Kafka: {log_data}")
#             try:
#                 # Assuming log_data is a dictionary structured like LogEntryDict.log_data
#                 # or a string that preprocess_input_log can handle
#                 prediction = predict_threat(log_data)
#                 logging.info(f"Kafka log classified: {prediction} - Log: {log_data}")
#                 # Here you could send the prediction to another topic, database, or monitoring system
#             except Exception as e:
#                 logging.error(f"Error processing Kafka message: {log_data} - Error: {e}")
#     except Exception as e:
#         logging.error(f"Kafka Consumer Error: {e}. Is Kafka running and topic '{KAFKA_TOPIC}' created?")

# @app.on_event("startup")
# async def startup_kafka_consumer():
#     # To run Kafka consumer in background (requires asyncio task management or separate thread/process)
#     # import asyncio
#     # logging.info("Starting Kafka consumer in background...")
#     # asyncio.create_task(consume_kafka_logs())
#     # For simplicity in this example, we won't auto-start it.
#     # You would typically use a library like 'aiokafka' for async FastAPI integration
#     # or run the consumer as a separate Python process.
#     pass

if __name__ == "__main__":
    import uvicorn

    # This is for local development/testing without Docker
    # For Docker, Gunicorn with Uvicorn workers is typically used via CMD in Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8000)