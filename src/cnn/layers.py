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
    

class ReLU(Layer):
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class MaxPool(Layer):
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        
        h_out = (H - self.pool_size) // self.stride + 1
        w_out = (W - self.pool_size) // self.stride + 1
        x_reshaped = x.reshape(N, C, h_out, self.stride, w_out, self.stride)
        out = x_reshaped.max(axis=3).max(axis=4)
        
        self.x_reshaped = x_reshaped
        self.out = out
        return out

    def backward(self, dout):
        dx_reshaped = xp.zeros_like(self.x_reshaped)
        
        out_newaxis = self.out[:, :, :, np.newaxis, :, np.newaxis]
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        

        mask = (self.x_reshaped == out_newaxis)       
        dx_reshaped = mask * dout_newaxis
        dx = dx_reshaped.reshape(self.x_shape)
        return dx
    
class Dense(Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        scale = xp.sqrt(2.0 / input_dim)
        self.params['w'] = xp.random.randn(input_dim, output_dim) * scale
        self.params['b'] = xp.zeros(output_dim)
        
        self.grads['w'] = None
        self.grads['b'] = None
        
    def forward(self, x):
        self.x_flat = x.reshape(x.shape[0], -1)
        out = xp.dot(self.x_flat, self.params['w']) + self.params['b']
        return out

    def backward(self, dout):
        self.grads['w'] = xp.dot(self.x_flat.T, dout)
        self.grads['b'] = xp.sum(dout, axis=0)
        dx_flat = xp.dot(dout, self.params['w'].T)
        return dx_flat.reshape(self.x_flat.shape[0], -1)
    

class SoftmaxCrossEntropy(Layer):
    def forward(self, x, y_true):
        self.x = x
        self.y_true = y_true
        N = x.shape[0]
        
        x_shifted = x - xp.max(x, axis=1, keepdims=True)
        exps = xp.exp(x_shifted)
        self.probs = exps / xp.sum(exps, axis=1, keepdims=True)
        
        correct_logprobs = -xp.log(self.probs[xp.arange(N), y_true] + 1e-7) # 1e-7 prevents log(0)
        loss = xp.sum(correct_logprobs) / N
        return loss

    def backward(self):
        N = self.x.shape[0]
        dx = self.probs.copy()

        dx[xp.arange(N), self.y_true] -= 1
        dx /= N
        return dx
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if not training:
            return x

        self.mask = (xp.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask
    

# wtf batchnorm hard af
class BatchNorm(Layer):
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters (Gamma: Scale, Beta: Shift)
        # Initialize Gamma to 1, Beta to 0
        self.params['gamma'] = xp.ones(num_features)
        self.params['beta'] = xp.zeros(num_features)
        
        # Gradients
        self.grads['gamma'] = None
        self.grads['beta'] = None
        
        # Running statistics (Non-learnable, for inference)
        # We store them in params for easy saving/loading, but optimizer ignores them
        self.params['running_mean'] = xp.zeros(num_features)
        self.params['running_var'] = xp.ones(num_features)

    def forward(self, x, training=True):
        self.x_shape = x.shape
        
        # Handle both FC (2D) and Conv (4D) inputs
        if x.ndim == 2:
            N, D = x.shape
            x_flat = x
        elif x.ndim == 4:
            N, C, H, W = x.shape
            # Reshape (N, C, H, W) -> (N*H*W, C) so we normalize per channel
            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
        
        if training:
            # 1. Calculate Mean and Variance
            mean = xp.mean(x_flat, axis=0)
            var = xp.var(x_flat, axis=0)
            
            # 2. Normalize
            x_hat = (x_flat - mean) / xp.sqrt(var + self.epsilon)
            
            # 3. Scale and Shift (y = gamma * x_hat + beta)
            out_flat = self.params['gamma'] * x_hat + self.params['beta']
            
            # 4. Update Running Statistics (Exponential Moving Average)
            self.params['running_mean'] = (self.momentum * self.params['running_mean']) + ((1 - self.momentum) * mean)
            self.params['running_var'] = (self.momentum * self.params['running_var']) + ((1 - self.momentum) * var)
            
            # Cache for backward pass
            self.cache = (x_flat, x_hat, mean, var)
        else:
            # Inference Mode: Use running stats
            x_hat = (x_flat - self.params['running_mean']) / xp.sqrt(self.params['running_var'] + self.epsilon)
            out_flat = self.params['gamma'] * x_hat + self.params['beta']

        # Reshape back to original dimensions
        if x.ndim == 4:
            N, C, H, W = self.x_shape
            out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            out = out_flat
            
        return out

    def backward(self, dout):
        # Retrieve cache
        x_flat, x_hat, mean, var = self.cache
        
        # Reshape dout if needed
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            dout_flat = dout
            
        N = dout_flat.shape[0]

        # 1. Gradient with respect to Gamma (Scale) and Beta (Shift)
        self.grads['gamma'] = xp.sum(dout_flat * x_hat, axis=0)
        self.grads['beta'] = xp.sum(dout_flat, axis=0)

        # 2. Gradient backprop through the normalization (The tricky part)
        # dL/dx_hat
        dx_hat = dout_flat * self.params['gamma']
        
        # Intermediate gradients
        ivar = 1.0 / xp.sqrt(var + self.epsilon)
        
        # dL/dVar
        dvar = xp.sum(dx_hat * (x_flat - mean) * -0.5 * (ivar**3), axis=0)
        
        # dL/dMean
        dmean = xp.sum(dx_hat * -ivar, axis=0) + dvar * xp.mean(-2.0 * (x_flat - mean), axis=0)
        
        # Final dL/dx
        dx_flat = (dx_hat * ivar) + (dvar * 2.0 * (x_flat - mean) / N) + (dmean / N)

        # Reshape dx back
        if len(self.x_shape) == 4:
            N, C, H, W = self.x_shape
            dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        else:
            dx = dx_flat
            
        return dx