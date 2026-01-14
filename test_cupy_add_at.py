import cupy as cp
import numpy as np

try:
    N, C, H, W = 2, 3, 4, 4
    x = cp.zeros((N, C, H, W))
    
    # Simulate indices
    # k: channels (C, 1) broadcasted
    k = cp.array([[0], [1], [2]]) 
    # i, j: pixel locs
    i = cp.array([[0, 1], [0, 1], [0, 1]])
    j = cp.array([[0, 1], [0, 1], [0, 1]])
    
    # Values to add
    # Shape matching (N, C, 2) roughly
    vals = cp.ones((N, 3, 2))
    
    # Broadcast simulation similar to im2col
    # Indices in im2col: (slice(None), k, i, j)
    # k is (Area*C, 1)
    # i is (Area*C, OutputPixels)
    
    # Let's match the broadcasting exactly or close enough
    # Target x: (2, 3, 4, 4)
    # Indices: (slice(None), k, i, j)
    # k shape: (3, 1)
    # i shape: (3, 2)
    # j shape: (3, 2)
    # result shape of indexing: (2, 3, 2) (N, k-dim, i-dim)
    
    # vals shape: (2, 3, 2)
    
    print("Testing cupy.add.at with slice...")
    cp.add.at(x, (slice(None), k, i, j), vals)
    print("Success!")
    print(x)
except Exception as e:
    print("Failed!")
    print(e)
