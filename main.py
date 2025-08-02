# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

# Define the class names based on your model's output order
# Ensure this matches the order your model was trained on!
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 256 # From your training.ipynb
MODEL_PATH = "exported_models/3" # Assuming your model is saved here. Adjust if necessary.

# Global variable to hold the loaded model
model = None

# Function to load the TensorFlow model
def load_tf_model():
    """Loads the pre-trained TensorFlow model from the specified path."""
    global model
    try:
        model = tf.saved_model.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}: {e}")
        # Exit if model loading fails, as the app cannot function without it
        raise RuntimeError(f"Failed to load TensorFlow model: {e}")

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager for managing application startup and shutdown events.
    This is the recommended way to handle startup/shutdown logic in FastAPI.
    """
    print("Application startup: Loading model...")
    load_tf_model() # Load the model when the app starts
    yield # Application runs
    print("Application shutdown: Cleaning up (if any)...")
    # You can add cleanup logic here if needed, e.g., closing database connections

# Initialize FastAPI app with the lifespan context manager
app = FastAPI(
    title="Potato Leaf Disease API",
    description="API for predicting potato leaf diseases from images.",
    lifespan=lifespan # Assign the lifespan manager
)

# Configure CORS to allow requests from your frontend (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, for development. In production, specify your frontend URL(s).
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for the request body when uploading base64 image
class ImageData(BaseModel):
    image: str # Base64 encoded image string

@app.post("/predict")
async def predict_disease(image_data: ImageData):
    """
    Predicts the disease type of a potato leaf from an uploaded image.
    The image should be sent as a base64 encoded string in the request body.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Decode the base64 image string
        # The frontend sends 'data:image/png;base64,...' or similar, so we split to get just the base64 part
        header, encoded = image_data.image.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        # Open the image using Pillow
        img = Image.open(io.BytesIO(image_bytes))

        # Ensure image is in RGB format (handle cases like RGBA or grayscale)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image to the model's expected input size
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        # Convert image to a NumPy array
        img_array = np.array(img)

        # Normalize the image pixels to [0, 1] as per your training
        img_array = img_array / 255.0

        # Add a batch dimension (model expects shape (batch_size, height, width, channels))
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction using the loaded model
        # Assuming the loaded model has a 'signatures' attribute for serving
        predictions = model.signatures["serving_default"](tf.constant(img_array, dtype=tf.float32))

        # The output tensor name might vary. Common ones are 'output_0' or 'dense_output'.
        # You might need to inspect your model's signature to get the correct output key.
        # For simplicity, let's assume the predictions are directly accessible as a tensor.
        # If predictions is a dict, you might need predictions['your_output_layer_name'].numpy()
        output_tensor_name = list(predictions.keys())[0] # Get the first output tensor name
        output_values = predictions[output_tensor_name].numpy()[0] # Get the numpy array from the tensor
        print(f"DEBUG: Raw model output values: {output_values}")
        print(f"DEBUG: Shape of output values: {output_values.shape}")

        predicted_class_index = np.argmax(output_values)
        confidence_score = float(output_values[predicted_class_index]) * 100

        predicted_disease = CLASS_NAMES[predicted_class_index]

        return JSONResponse(content={
            "disease": predicted_disease,
            "confidence": round(confidence_score, 2)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Basic root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Potato Leaf Disease API is running!"}


    #To run the model paste this line in terminal and double click on index.html to upload image 
    #->uvicorn main:app --reload --host 0.0.0.0 --port 8000
