import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = './ml_model/saved_transformer/'
MODEL_NAME = 'distilbert-base-uncased' # Should match training
MAX_LENGTH = 128 # Should match training

# These features must match those used during training for consistent input string construction
# This is critical if the input is a JSON object of features
FEATURE_COLUMNS_ORDERED = [
    'srcip', 'sport', 'dstip', 'dport', 'proto', 'state',
    'service', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
    'Sload', 'Dload', 'Spkts', 'Dpkts'
] # Ensure this order is consistent if reconstructing from dict

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        logging.info(f"Loading model and tokenizer from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
            logging.error(f"Model directory {MODEL_PATH} is empty or does not exist.")
            logging.error("Please train the model first using 'python ml_model/train_model.py'")
            raise FileNotFoundError("Model not found. Train the model first.")
        try:
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
            model.eval() # Set to evaluation mode
            logging.info("Model and tokenizer loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model/tokenizer: {e}")
            raise
    return model, tokenizer

def preprocess_input_log(log_data):
    """
    Preprocesses a single log entry.
    log_data can be:
    1. A raw string: "srcip:1.2.3.4 sport:123 ..."
    2. A dictionary: {"srcip": "1.2.3.4", "sport": "123", ...}
    """
    if isinstance(log_data, dict):
        # Construct the string from dict, ensuring correct order and format
        # Fill missing keys with a default value like 'unknown' or 0
        # This must match the string format used in training's preprocess_data
        log_text_parts = []
        for col in FEATURE_COLUMNS_ORDERED:
            value = log_data.get(col, 'unknown') # or 0 if numerical and filled with 0 in training
            log_text_parts.append(f"{col}:{str(value)}")
        log_text = " ".join(log_text_parts)
    elif isinstance(log_data, str):
        log_text = log_data # Assume it's already in "feature:value feature:value" format
    else:
        raise ValueError("Input log_data must be a string or a dictionary.")
    return log_text

def predict_threat(log_input_data):
    """
    Predicts if a log entry is malicious or normal.
    log_input_data: Raw log string or a dictionary of features.
    """
    try:
        _model, _tokenizer = load_model()
    except FileNotFoundError:
        return "Error: Model not found. Please train or ensure it's correctly placed."
    except Exception as e:
        return f"Error loading model: {str(e)}"

    processed_log_text = preprocess_input_log(log_input_data)
    logging.info(f"Processed log for prediction: {processed_log_text}")

    inputs = _tokenizer(
        processed_log_text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = _model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction_idx = torch.argmax(logits, dim=1).item()

    # Assuming 0: normal, 1: malicious (consistent with training)
    return "malicious" if prediction_idx == 1 else "normal"

if __name__ == '__main__':
    # Quick test
    print("Testing model_utils.py...")
    # Ensure the model is trained and saved before running this test
    if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
        print(f"Skipping test: Model not found at {MODEL_PATH}. Please train the model first.")
    else:
        print(f"Attempting to load model from {MODEL_PATH}...")
        try:
            # Example log string (adjust features based on your FEATURE_COLUMNS)
            sample_log_str = "srcip:192.168.0.1 sport:12345 dstip:10.0.0.1 dport:80 proto:tcp state:SYN service:http sbytes:60 dbytes:0 sttl:64 dttl:0 sloss:0 dloss:0 Sload:8000 Dload:0 Spkts:1 Dpkts:0"
            prediction_str = predict_threat(sample_log_str)
            print(f"Prediction for sample log string: '{sample_log_str}' -> {prediction_str}")

            sample_log_dict = {
                "srcip": "149.171.126.0", "sport": "61105", "dstip": "175.45.176.0", "dport": "80",
                "proto": "tcp", "state": "FIN", "service": "http", "sbytes": "2720",
                "dbytes": "2902", "sttl": "31", "dttl": "29", "sloss": "0", "dloss": "0",
                "Sload": "889118.062500", "Dload": "22270.207030", "Spkts": "14", "Dpkts": "12"
            }
            prediction_dict = predict_threat(sample_log_dict)
            print(f"Prediction for sample log dict: {sample_log_dict} -> {prediction_dict}")

        except Exception as e:
            print(f"Error during test: {e}")