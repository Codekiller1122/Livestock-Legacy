from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image
from typing import Dict

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define constants
IMAGE_SIZE = 256  # Update if different
class_names = ['healthy', 'lumpy']  # Update with actual class names

# Load the saved model
model_path = 'D:/new_cow_disease_CNN.h5'
model = tf.keras.models.load_model(model_path, compile=False)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Function to predict new images
def predict(model, img):
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch axis
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    # Convert numpy.float32 to a standard Python float
    confidence = float(np.max(predictions[0]))  # Convert to standard float
    return predicted_class, confidence

# Endpoint to receive image and return prediction
@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(BytesIO(contents))

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    # Resize image
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

    predicted_class, confidence = predict(model, img)
    
    return {"predicted_class": predicted_class, "confidence": confidence}
