import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.loader import load_mnist
from src.cnn.layers import ConvLayer, ReLU, MaxPool, Dense, SoftmaxCrossEntropy
from src.cnn.network import Sequential
from src.cnn.optimizers import Adam
from src.utils.evaluation import evaluate_model

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
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate timestamped filename
    import time
    timestamp = int(time.time())
    model_name = f"mnist_model_{timestamp}.pkl"
    save_path = os.path.join(models_dir, model_name)
    
    print(f"Saving model to {save_path}...")
    model.save(save_path)

    # 7. Detailed Evaluation
    print("Generating detailed classification report and heatmap...")
    y_pred = model.predict(x_test, batch_size=64)
    
    # Save reports to reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    evaluate_model(y_test, y_pred, class_names=[str(i) for i in range(10)], save_dir=reports_dir, prefix=f'mnist_train_{timestamp}')

if __name__ == "__main__":
    main()