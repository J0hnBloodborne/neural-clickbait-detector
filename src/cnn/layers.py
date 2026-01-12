# Convolution, ReLU, MaxPool, FC layer classes
import numpy as np
from src.cnn.im2col import im2col_indices, get_im2col_indices, col2im_indices

xp = np
def use_gpu():
    global xp
    try:
        import cupy
        xp = cupy
        return True
    except ImportError:
        return False
class Layer:
    def __init__(self):
        self.params = {} # w, b
        self.grads = {}  # dw, db

    def forward(self, x):
        pass

    def backward(self, dout):
        pass


class ConvLayer(Layer):
    def __init__(self, num_filters, input_channels, filter_size, stride=1, padding=1):
        super().__init__()
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding

        fan_in = input_channels * filter_size * filter_size

        scale = xp.sqrt(2.0 / fan_in)

        self.params['w'] = xp.random.randn(num_filters, input_channels, filter_size, filter_size) * scale

        self.params['b'] = xp.zeros(num_filters)
        
        self.grads['w'] = None
        self.grads['b'] = None

    def forward(self, x):
        # save shape for backprop
        self.x_shape = x.shape
        N, C, H, W = x.shape
        
        # 1. Turn image into columns
        self.x_cols = im2col_indices(x, self.filter_size, self.filter_size, self.padding, self.stride)
        
        # 2. Flatten weights to rows
        w_row = self.params['w'].reshape(self.num_filters, -1)
        
        # 3. The Big Matrix Multiplication
        out = xp.dot(w_row, self.x_cols) + self.params['b'].reshape(-1, 1)
        
        # 4. Reshape back to image dimensions
        h_out = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # 5. Reshape to (NumFilters, H, W, Batch) then transpose to (Batch, NumFilters, H, W)
        out = out.reshape(self.num_filters, h_out, w_out, N)
        out = out.transpose(3, 0, 1, 2)
        
        return out
    
    def backward(self, dout):

        # 1. Reshape dout to match the matrix multiplication format
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)
        
        # 2. dW = dout @ x_cols.T
        dw = xp.dot(dout_reshaped, self.x_cols.T)
        self.grads['w'] = dw.reshape(self.params['w'].shape)
        
        # 3. db = sum(dout)
        self.grads['b'] = xp.sum(dout, axis=(0, 2, 3))
        
        # 4. dX_cols = W.T @ dout
        w_flat = self.params['w'].reshape(self.num_filters, -1)
        dx_cols = xp.dot(w_flat.T, dout_reshaped)
        
        # 5. Fold dX_cols back into image shape (Inverse of im2col)
        dx = col2im_indices(dx_cols, self.x_shape, self.filter_size, self.filter_size, self.padding, self.stride)
        
        return dx