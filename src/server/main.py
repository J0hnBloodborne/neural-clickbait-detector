import sys
import os
import io
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add root to path so we can import src.cnn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.cnn.network import Sequential
from src.cnn.layers import * # Needed for pickle loading
from src.cnn.optimizers import * # Needed for pickle loading

app = FastAPI()

# Setup Templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load Model (Global variable)
model = Sequential()
MODEL_PATH = os.path.join(os.getcwd(), "mnist_model.pkl")

# Attempt to load model on startup
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model.layers = [
        ConvLayer(32, 1, 3, padding=1), ReLU(), MaxPool(2, 2),
        ConvLayer(64, 32, 3, padding=1), ReLU(), MaxPool(2, 2),
        Dense(64 * 7 * 7, 128), ReLU(),
        Dense(128, 10)
    ]
    # We load params into the structure defined above
    model.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    print(f"Warning: {MODEL_PATH} not found. Please run train_mnist.py first.")

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float

def process_image(image_bytes):
    """
    Standardize image: Grayscale -> Crop Whitespace -> Resize 28x28 -> Normalize
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    # Invert if needed (Canvas is black on white, MNIST is white on black)
    # We assume user draws Black on White canvas, so we invert.
    img = ImageOps.invert(img)
    
    # 1. Crop Content (Reuse your logic)
    img_array = np.array(img)
    rows = np.where(np.max(img_array, axis=1) > 0)[0]
    cols = np.where(np.max(img_array, axis=0) > 0)[0]
    
    if len(rows) > 0 and len(cols) > 0:
        top, bottom = rows[0], rows[-1] + 1
        left, right = cols[0], cols[-1] + 1
        # Add padding
        pad = 4
        img = img.crop((
            max(0, left - pad), 
            max(0, top - pad), 
            min(img.width, right + pad), 
            min(img.height, bottom + pad)
        ))

    # 2. Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 3. To Numpy (N, C, H, W)
    x = np.array(img).astype(np.float32) / 255.0
    x = x.reshape(1, 1, 28, 28)
    
    return x

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model.layers:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    image_bytes = await file.read()
    x = process_image(image_bytes)
    
    # Inference
    logits = model.forward(x, training=False)
    
    # Softmax for confidence
    exps = np.exp(logits - np.max(logits))
    probs = exps / np.sum(exps)
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[0][pred_idx])
    
    return {"prediction": pred_idx, "confidence": confidence}