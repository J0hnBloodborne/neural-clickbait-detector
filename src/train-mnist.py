import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_mnist
from src.cnn.layers import ConvLayer, ReLU, MaxPool, Dense, SoftmaxCrossEntropy
from src.cnn.network import Sequential
from src.cnn.optimizers import Adam

# Import modules to enable GPU
import src.cnn.layers
import src.cnn.network
import src.cnn.optimizers
import src.cnn.im2col

def main():
    print("--- MNIST Training Script ---")
    
    # 1. Load Data (Parses local .idx3-ubyte files)
    print("Loading data...")
    try:
        (x_train, y_train), (x_test, y_test) = load_mnist()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Check that your data/mnist folder contains the unzipped .ubyte files.")
        return

    # 2. Build Model
    # Input: 1 x 28 x 28
    model = Sequential([
        # Layer 1: Conv 32 filters
        ConvLayer(32, 1, 3, padding=1),
        ReLU(),
        MaxPool(2, 2), # -> 14x14
        
        # Layer 2: Conv 64 filters
        ConvLayer(64, 32, 3, padding=1),
        ReLU(),
        MaxPool(2, 2), # -> 7x7
        
        # Layer 3: Fully Connected
        Dense(64 * 7 * 7, 128),
        ReLU(),
        Dense(128, 10)
    ])
    
    # 3. Compile
    model.compile(loss=SoftmaxCrossEntropy(), optimizer=Adam(lr=0.001))

    # GPU Transfer
    try:
        import cupy
        # Enable GPU in modules
        print("Enabling GPU in library modules...")
        src.cnn.im2col.use_gpu()
        src.cnn.layers.use_gpu()
        src.cnn.network.use_gpu()
        src.cnn.optimizers.use_gpu()

        print("GPU Detected. Moving data to VRAM...")
        x_train = cupy.asarray(x_train)
        y_train = cupy.asarray(y_train)
        x_test = cupy.asarray(x_test)
        y_test = cupy.asarray(y_test)
        
        print("Moving model weights to VRAM...")
        for layer in model.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    layer.params[key] = cupy.asarray(layer.params[key])
    except ImportError:
        print("No GPU detected. Training on CPU (this will be slower).")
    
    # 4. Train
    print("Starting training loop...")
    model.fit(x_train, y_train, epochs=5, batch_size=64)
    
    # 5. Evaluate
    print("\nEvaluating on Test Set:")
    model.evaluate(x_test, y_test)
    
    # 6. Save
    model.save("mnist_model.pkl")

if __name__ == "__main__":
    main()