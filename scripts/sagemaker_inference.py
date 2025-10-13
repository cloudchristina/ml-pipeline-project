"""
SageMaker inference script for sentiment analysis model.
This script handles model loading and prediction requests.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global variables (loaded once when container starts)
tokenizer = None
model = None
device = None


def model_fn(model_dir):
    """
    Load the model from the model_dir.
    Called once when the container starts.

    Args:
        model_dir: Path to the model artifacts

    Returns:
        dict: Dictionary containing model and tokenizer
    """
    global tokenizer, model, device

    print(f"Loading model from {model_dir}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()  # Set to evaluation mode

    print("Model loaded successfully")

    return {"model": model, "tokenizer": tokenizer, "device": device}


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.

    Args:
        request_body: The request payload
        request_content_type: The content type of the request

    Returns:
        dict: Parsed input data
    """
    print(f"Received content type: {request_content_type}")

    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """
    Perform prediction on the input data.

    Args:
        input_data: Preprocessed input
        model_dict: Dictionary containing model and tokenizer

    Returns:
        dict: Prediction results
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    device = model_dict["device"]

    # Extract text from input
    if isinstance(input_data, dict):
        text = input_data.get("text", "")
    elif isinstance(input_data, str):
        text = input_data
    else:
        raise ValueError("Input must be a dict with 'text' key or a string")

    if not text:
        raise ValueError("Empty text provided")

    print(f"Processing text: {text[:100]}...")  # Log first 100 chars

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Map class to sentiment
    sentiment = "positive" if predicted_class == 1 else "negative"

    result = {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "negative": round(probabilities[0][0].item(), 4),
            "positive": round(probabilities[0][1].item(), 4)
        }
    }

    print(f"Prediction: {result}")

    return result


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output.

    Args:
        prediction: The prediction result
        response_content_type: The desired response content type

    Returns:
        str: Serialized prediction
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
