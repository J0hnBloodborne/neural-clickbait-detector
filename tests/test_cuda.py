import numpy as np
import time

def test_cuda_setup():
    print("--- CUDA / CuPy Environment Check ---")
    
    try:
        import cupy as cp
        print(f"CuPy found. Version: {cp.__version__}")
        
        dev = cp.cuda.Device()
        print(f"GPU Device: {dev.id} (Compute Capability: {dev.compute_capability})")
        
    except ImportError:
        print("CuPy NOT found. You are strictly on CPU mode.")
        print("To install: pip install cupy-cuda11x (check your CUDA version)")
        return

    print("\n--- Functional Test ---")
    # 1. Create array on CPU
    x_cpu = np.array([1, 2, 3])
    print(f"1. Created CPU array: {x_cpu}")
    
    # 2. Move to GPU
    x_gpu = cp.asarray(x_cpu)
    print(f"2. Moved to GPU: {x_gpu} (Type: {type(x_gpu)})")
    
    # 3. Operation on GPU
    x_gpu = x_gpu * 2
    print(f"3. Multiplied on GPU: {x_gpu}")
    
    # 4. Move back to CPU
    x_back = cp.asnumpy(x_gpu)
    print(f"4. Moved back to CPU: {x_back} (Type: {type(x_back)})")
    
    assert np.all(x_back == [2, 4, 6]), "Math error on GPU."
    print("Logic check passed.")

    print("\n--- Speed Test: Matrix Multiplication (4000x4000) ---")
    
    size = 4000
    
    # CPU Test
    print("Generating random matrices on CPU...")
    a_cpu = np.random.rand(size, size)
    b_cpu = np.random.rand(size, size)
    
    print("Running CPU Dot Product...")
    start = time.time()
    np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # GPU Test
    print("\nMoving data to GPU...")
    a_gpu = cp.asarray(a_cpu)
    b_gpu = cp.asarray(b_cpu)
    
    # Warmup
    cp.dot(a_gpu[:10, :10], b_gpu[:10, :10])
    cp.cuda.Stream.null.synchronize()
    
    print("Running GPU Dot Product...")
    start = time.time()
    cp.dot(a_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize() 
    gpu_time = time.time() - start
    
    print(f"GPU Time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

if __name__ == "__main__":
    test_cuda_setup()