import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_cifar10_kaggle
from src.cnn.layers import ConvLayer, ReLU, MaxPool, Dense, SoftmaxCrossEntropy, BatchNorm, Dropout
from src.cnn.layers import use_gpu as use_gpu_layers
from src.cnn.im2col import use_gpu as use_gpu_im2col
from src.cnn.network import Sequential, use_gpu as use_gpu_network
from src.cnn.optimizers import Adam, use_gpu as use_gpu_optimizers

def main():
    print("--- CIFAR-10 Training Script ---")

    # 1. Load Data (Parses Images + CSV)
    # Set limit=1000 for a quick debug run. Set limit=None for full training.
    print("Loading data from Kaggle folder...")
    try:
        # We split the Kaggle 'train' folder into Train/Val because Kaggle 'test' has no labels
        (x_train, y_train), (x_val, y_val) = load_cifar10_kaggle(limit=None)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure 'trainLabels.csv' and the 'train' folder are in data/cifar-10/")
        return

    # 2. Build VGG-Style Model
    # Input: 3 x 32 x 32
    model = Sequential([
        # Block 1
        ConvLayer(32, 3, 3, padding=1),
        BatchNorm(32),
        ReLU(),
        ConvLayer(32, 32, 3, padding=1),
        BatchNorm(32),
        ReLU(),
        MaxPool(2, 2),
        Dropout(0.25),
        
        # Block 2
        ConvLayer(64, 32, 3, padding=1),
        BatchNorm(64),
        ReLU(),
        ConvLayer(64, 64, 3, padding=1),
        BatchNorm(64),
        ReLU(),
        MaxPool(2, 2),
        Dropout(0.25),
        
        # Block 3
        ConvLayer(128, 64, 3, padding=1),
        BatchNorm(128),
        ReLU(),
        MaxPool(2, 2),
        Dropout(0.25),
        
        # Classifier
        Dense(128 * 4 * 4, 128),
        BatchNorm(128),
        ReLU(),
        Dropout(0.5),
        Dense(128, 10)
    ])
    
    # 3. Compile
    model.compile(loss=SoftmaxCrossEntropy(), optimizer=Adam(lr=0.001))

    # GPU Transfer
    try:
        import cupy
        print("GPU Detected. Moving data to VRAM...")
        
        # Enable GPU in all modules
        use_gpu_layers()
        use_gpu_im2col()
        use_gpu_network()
        use_gpu_optimizers()

        x_train = cupy.asarray(x_train)
        y_train = cupy.asarray(y_train)
        x_val = cupy.asarray(x_val)
        y_val = cupy.asarray(y_val)
        
        print("Moving model weights to VRAM...")
        for layer in model.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    layer.params[key] = cupy.asarray(layer.params[key])

    except ImportError:
        print("No GPU detected. Training on CPU will be very slow for CIFAR.")
    
    # 4. Train
    # CIFAR is harder, needs more epochs.
    print(f"Starting training on {x_train.shape[0]} samples...")
    model.fit(x_train, y_train, epochs=30, batch_size=128, x_val=x_val, y_val=y_val, patience=5)
    
    # 5. Evaluate
    print("\nEvaluating on Validation Split:")
    model.evaluate(x_val, y_val, batch_size=64)
    
    # 6. Save
    model.save("cifar_vgg.pkl")

if __name__ == "__main__":
    main()