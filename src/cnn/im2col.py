import numpy as np

# Default to NumPy (CPU)
xp = np 

# Function to toggle GPU
def use_gpu():
    global xp
    try:
        import cupy
        xp = cupy
        print("Switched to GPU (CuPy)!")
        return True
    except ImportError:
        print("CuPy not found. Staying on CPU.")
        return False

def to_cpu(x):
    if xp == np: return x
    return xp.asnumpy(x)

# padding = 0 -> valid
# padding = 1 -> same
# padding > 1 -> full
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # calc output dims
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    # inner offsets
    i0 = xp.repeat(xp.arange(field_height), field_width)
    i0 = xp.tile(i0, C)
    j0 = xp.tile(xp.arange(field_width), field_height * C)

    # outer offsets
    i1 = stride * xp.repeat(xp.arange(out_height), out_width)
    j1 = stride * xp.tile(xp.arange(out_width), out_height)

    # final coords
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):

    x_padded = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    # get indices and pixels
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    
    # reshape into column
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    
    x_padded = xp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    xp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0: return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]