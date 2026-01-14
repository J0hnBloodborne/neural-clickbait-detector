import numpy as np

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

class Optimizer:
    def step(self, layers):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {} 

    def step(self, layers):
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'params'):
                continue
            
            for key in layer.params.keys():
                w = layer.params[key]
                dw = layer.grads[key]
                
                # Create unique key for this parameter
                v_key = f"{i}_{key}"
                if v_key not in self.velocities:
                    self.velocities[v_key] = xp.zeros_like(w)
                
                # v = mu * v - lr * dw
                self.velocities[v_key] = (self.momentum * self.velocities[v_key]) - (self.lr * dw)
                
                # w = w + v
                layer.params[key] += self.velocities[v_key]

class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {} # First moment
        self.v = {} # Second moment
        self.t = 0  # Time step

    def step(self, layers):
        self.t += 1
        
        for i, layer in enumerate(layers):
            if not hasattr(layer, 'params'):
                continue

            for key in layer.params.keys():
                # Skip parameters without gradients (e.g. running_mean)
                if key not in layer.grads or layer.grads[key] is None:
                    continue

                # Get parameter and gradient
                w = layer.params[key]
                dw = layer.grads[key]
                
                # Unique ID for state tracking
                p_key = f"{i}_{key}"
                
                # Initialize state if not exists
                if p_key not in self.m:
                    self.m[p_key] = xp.zeros_like(w)
                    self.v[p_key] = xp.zeros_like(w)
                
                # 1. Update biased first moment estimate
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g
                self.m[p_key] = self.beta1 * self.m[p_key] + (1 - self.beta1) * dw
                
                # 2. Update biased second raw moment estimate
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
                self.v[p_key] = self.beta2 * self.v[p_key] + (1 - self.beta2) * (dw ** 2)
                
                # 3. Compute bias-corrected first moment estimate
                # m_hat = m_t / (1 - beta1^t)
                m_hat = self.m[p_key] / (1 - self.beta1 ** self.t)
                
                # 4. Compute bias-corrected second raw moment estimate
                # v_hat = v_t / (1 - beta2^t)
                v_hat = self.v[p_key] / (1 - self.beta2 ** self.t)
                
                # 5. Update parameters
                # w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
                layer.params[key] -= self.lr * m_hat / (xp.sqrt(v_hat) + self.epsilon)