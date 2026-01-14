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

# Setup Templates (Define BASE_DIR first)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global State
model = Sequential()
current_model_type = "mnist" # or 'cifar'
MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), 'models'))

def get_available_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    files = os.listdir(MODELS_DIR)
    models = []
    for f in files:
        if f.endswith('.pkl'):
            # Detect type from filename
            m_type = 'cifar' if 'cifar' in f else 'mnist'
            # Format name nicely: "mnist_model_123.pkl" -> "MNIST (123)"
            name_parts = f.replace('.pkl', '').split('_')
            ts = name_parts[-1]
            label = f"{m_type.upper()} ({ts})"
            models.append({"name": label, "filename": f, "type": m_type})
            
    # Sort by newness (timestamp in filename is reliable for sorting)
    models.sort(key=lambda x: x['filename'], reverse=True)
    return models

@app.get("/models")
def list_models():
    return get_available_models()

@app.get("/load_model")
def load_model_endpoint(model: str):
    global current_model_type
    path = os.path.join(MODELS_DIR, model)
    if not os.path.exists(path):
        return {"error": "Model not found"}
        
    try:
        print(f"Loading model from {path}...")
        # We need to reconstruct the architecture before loading params
        # This is a limitation of not saving architecture in the pkl (usually)
        # But if we pickle the whole object, we can just load it?
        # Your save method in network.py saves 'params' dictionary, not the object.
        # So we must recreate structure.
        
        # Determine type
        if 'cifar' in model:
            current_model_type = 'cifar'
            # Recreate VGG structure from train-cifar.py
            main_network = Sequential([
                ConvLayer(32, 3, 3, padding=1), BatchNorm(32), ReLU(),
                ConvLayer(32, 32, 3, padding=1), BatchNorm(32), ReLU(),
                MaxPool(2, 2), Dropout(0.25),
                
                ConvLayer(64, 32, 3, padding=1), BatchNorm(64), ReLU(),
                ConvLayer(64, 64, 3, padding=1), BatchNorm(64), ReLU(),
                MaxPool(2, 2), Dropout(0.25),
                
                ConvLayer(128, 64, 3, padding=1), BatchNorm(128), ReLU(),
                MaxPool(2, 2), Dropout(0.25),
                
                Dense(128 * 4 * 4, 128), BatchNorm(128), ReLU(),
                Dropout(0.5), Dense(128, 10)
            ])
        else:
            current_model_type = 'mnist'
            # Recreate MNIST structure
            main_network = Sequential([
                ConvLayer(32, 1, 3, padding=1), ReLU(), MaxPool(2, 2),
                ConvLayer(64, 32, 3, padding=1), ReLU(), MaxPool(2, 2),
                Dense(64 * 7 * 7, 128), ReLU(),
                Dense(128, 10)
            ])
            
        main_network.load(path)
        
        # Update global model reference
        # We need to update the global 'model' variable's layers to match the new loaded one
        # Or just replace the object (but 'model' is imported as Sequential instance?)
        # Let's just swap globals
        globals()['model'] = main_network
        
        return {"status": "success", "type": current_model_type}
    except Exception as e:
        print(e)
        return {"error": str(e)}

def process_image(image_bytes, model_type):
    img = Image.open(io.BytesIO(image_bytes))
    
    if model_type == 'mnist':
        # MNIST Processing
        img = img.convert('L')
        img = ImageOps.invert(img) # Black ink -> White ink
        
        # Crop & Resize logic
        img_array = np.array(img)
        rows = np.where(np.max(img_array, axis=1) > 0)[0]
        cols = np.where(np.max(img_array, axis=0) > 0)[0]
        if len(rows) > 0 and len(cols) > 0:
            top, bottom = rows[0], rows[-1] + 1
            left, right = cols[0], cols[-1] + 1
            pad = 4
            img = img.crop((max(0, left-pad), max(0, top-pad), min(img.width, right+pad), min(img.height, bottom+pad)))
            
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        x = np.array(img).astype(np.float32) / 255.0
        x = x.reshape(1, 1, 28, 28)
        
    else: # CIFAR
        # CIFAR Processing (Color, 32x32)
        img = img.convert('RGB')
        img = img.resize((32, 32), Image.Resampling.LANCZOS)
        x = np.array(img).astype(np.float32) / 255.0
        # Transpose to (C, H, W) -> (3, 32, 32)
        x = x.transpose(2, 0, 1)
        x = x.reshape(1, 3, 32, 32)
        
    return x

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model.layers:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    image_bytes = await file.read()
    
    try:
        x = process_image(image_bytes, current_model_type)
        
        # Inference
        logits = model.forward(x, training=False)
        
        # Softmax for confidence
        # Logits shape (1, 10)
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        
        # Handle Cupy/Numpy duality
        if hasattr(probs, 'get'):
             probs = probs.get()
        
        probs = probs.flatten().tolist()
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        return {
            "prediction": pred_idx, 
            "confidence": confidence,
            "probabilities": probs
        }
    except Exception as e:
        print(e)
        return JSONResponse(status_code=500, content={"error": str(e)})