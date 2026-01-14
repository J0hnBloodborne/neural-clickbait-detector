# Assembled CNN class
import pickle
import numpy as np
import time
import sys

# Backend Setup
xp = np
def use_gpu():
    global xp
    try:
        import cupy
        xp = cupy
        return True
    except ImportError:
        return False

class Sequential:
    def __init__(self, layers=None):
        self.layers = layers if layers else []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer):
        self.loss_function = loss
        self.optimizer = optimizer

    def forward(self, x, training=True):
        out = x
        for layer in self.layers:
            if hasattr(layer, 'forward') and layer.forward.__code__.co_argcount > 2:
                 out = layer.forward(out, training=training)
            else:
                 out = layer.forward(out)
        return out

    def backward(self):
        dout = self.loss_function.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def predict(self, x):
        logits = self.forward(x, training=False)
        return xp.argmax(logits, axis=1)

    def fit(self, x_train, y_train, epochs=10, batch_size=32, verbose=True):
        if not self.optimizer or not self.loss_function:
            raise ValueError("Model not compiled. Call .compile() first.")

        num_samples = x_train.shape[0]
        num_batches = int(np.ceil(num_samples / batch_size))

        print(f"Training on {num_samples} samples ({num_batches} batches/epoch)")

        for epoch in range(epochs):
            total_loss = 0.0
            start_time = time.time()
            
            # Shuffle data
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, num_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Move to GPU if active
                if xp != np:
                    x_batch = xp.asarray(x_batch)
                    y_batch = xp.asarray(y_batch)

                # Forward
                logits = self.forward(x_batch, training=True)
                
                # Loss
                loss = self.loss_function.forward(logits, y_batch)
                total_loss += float(loss)

                # Backward
                self.backward()

                # Optimize
                self.optimizer.step(self.layers)

                # Simple inline progress update
                if verbose and b % 10 == 0:
                    sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} | Batch {b}/{num_batches} | Loss: {loss:.4f}")
                    sys.stdout.flush()

            # End of epoch summary
            avg_loss = total_loss / num_batches
            duration = time.time() - start_time
            if verbose:
                print(f"\rEpoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Time: {duration:.2f}s        ")

    def evaluate(self, x_test, y_test):
        if xp != np:
            x_test = xp.asarray(x_test)
            y_test = xp.asarray(y_test)
            
        preds = self.predict(x_test)
        accuracy = xp.mean(preds == y_test)
        
        if hasattr(accuracy, 'item'): 
            accuracy = accuracy.item()
            
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save(self, filename):
        # Serialize model parameters to CPU-compatible format
        params = []
        for layer in self.layers:
            p = {}
            if hasattr(layer, 'params'):
                for k, v in layer.params.items():
                    # Check if CuPy array and convert to NumPy
                    if hasattr(v, 'get'): 
                        p[k] = v.get()
                    else:
                        p[k] = v
            params.append(p)
            
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        print(f"Model saved to {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            params = pickle.load(f)
            
        for layer, p in zip(self.layers, params):
            if hasattr(layer, 'params'):
                for k, v in p.items():
                    # Load into current backend (NumPy or CuPy)
                    layer.params[k] = xp.array(v)
        print(f"Model loaded from {filename}")