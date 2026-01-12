import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cnn.im2col import im2col_indices

def test_im2col_logic():
    print("--- Testing im2col Logic ---")

    # 1. Create a dummy image (1 Batch, 1 Channel, 4 Height, 4 Width)
    # Values:
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11],
    #  [12, 13, 14, 15]]
    x = np.arange(16).reshape(1, 1, 4, 4)
    print("Original Image (4x4):")
    print(x[0, 0])

    # 2. Run im2col
    # Filter: 2x2, Stride: 2, Padding: 0
    # Expected behavior: No overlap.
    # Window 1 (Top-Left): [[0, 1], [4, 5]]
    # Window 2 (Top-Right): [[2, 3], [6, 7]]
    # Window 3 (Bot-Left): [[8, 9], [12, 13]]
    # Window 4 (Bot-Right): [[10, 11], [14, 15]]
    
    cols = im2col_indices(x, field_height=2, field_width=2, padding=0, stride=2)
    
    print("\nim2col Output (Cols):")
    print(cols)

    # 3. Check Dimensions
    # Input: (1, 1, 4, 4)
    # Output Rows: Channels(1) * FilterH(2) * FilterW(2) = 4
    # Output Cols: Windows(4) * Batch(1) = 4
    assert cols.shape == (4, 4), f"Shape mismatch: {cols.shape}"
    
    # 4. Check Values
    # Column 0 should be the flattened first window: [0, 1, 4, 5]
    expected_col0 = np.array([0, 1, 4, 5])
    if np.array_equal(cols[:, 0], expected_col0):
        print("\nPASS: Column 0 matches top-left window.")
    else:
        print(f"\nFAIL: Column 0 is {cols[:, 0]}, expected {expected_col0}")

    print("im2col visual check complete.")

if __name__ == "__main__":
    test_im2col_logic()