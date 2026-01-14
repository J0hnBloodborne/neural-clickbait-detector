import numpy as np
import sys
import os

# Path hack to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cnn.layers import ConvLayer

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        
        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval
        
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad

def test_conv_gradient():
    print("--- Testing Convolution Gradient ---")
    
    # Setup random inputs
    x = np.random.randn(2, 3, 5, 5).astype(np.float64)
    dout = np.random.randn(2, 2, 5, 5).astype(np.float64) # Output grad
    
    # Initialize Layer
    # Filters: 2, Channels: 3, Size: 3x3, Pad: 1
    conv = ConvLayer(num_filters=2, input_channels=3, filter_size=3, padding=1)
    
    # 1. Forward Pass
    out = conv.forward(x)
    
    # 2. Backward Pass (Analytic Gradient)
    dx_num = conv.backward(dout)
    dw_analytic = conv.grads['w']
    db_analytic = conv.grads['b']
    
    # 3. Numerical Gradient (Weights)
    def fw(w):
        conv.params['w'] = w
        return conv.forward(x)

    print("Checking Weights (dW)...")
    dw_num = eval_numerical_gradient_array(fw, conv.params['w'], dout)
    error = rel_error(dw_analytic, dw_num)
    print(f"Error: {error:.2e}")
    assert error < 1e-7, "Weights gradient check failed!"

    # 4. Numerical Gradient (Bias)
    def fb(b):
        conv.params['b'] = b
        return conv.forward(x)
        
    print("Checking Bias (db)...")
    db_num = eval_numerical_gradient_array(fb, conv.params['b'], dout)
    error = rel_error(db_analytic, db_num)
    print(f"Error: {error:.2e}")
    assert error < 1e-7, "Bias gradient check failed!"

    print("Gradient check passed.")

if __name__ == "__main__":
    test_conv_gradient()