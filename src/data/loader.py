import numpy as np
import os
import struct
import csv
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_mnist():
    """
    Parses the Yann LeCun IDX3-UBYTE format manually.
    """
    mnist_dir = os.path.join(DATA_DIR, 'mnist')
    
    files = {
        'train_img': 'train-images.idx3-ubyte',
        'train_lbl': 'train-labels.idx1-ubyte',
        'test_img': 't10k-images.idx3-ubyte',
        'test_lbl': 't10k-labels.idx1-ubyte'
    }

    def get_path(fname):
        p1 = os.path.join(mnist_dir, fname)
        p2 = os.path.join(mnist_dir, fname.replace('.', '-')) # Handle train-images-idx3...
        if os.path.exists(p1): return p1
        if os.path.exists(p2): return p2
        raise FileNotFoundError(f"Could not find MNIST file: {fname} in {mnist_dir}")

    def parse_idx(filename):
        with open(filename, 'rb') as f:
            # Read Magic Number and Count
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            
            # Read data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(shape)

    print("Parsing MNIST binary files...")
    x_train = parse_idx(get_path('train-images.idx3-ubyte'))
    y_train = parse_idx(get_path('train-labels.idx1-ubyte'))
    x_test = parse_idx(get_path('t10k-images.idx3-ubyte'))
    y_test = parse_idx(get_path('t10k-labels.idx1-ubyte'))

    # Normalize and Reshape to (N, 1, 28, 28)
    x_train = x_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def load_cifar10_kaggle(limit=None):
    """
    Parses the Kaggle CIFAR-10 version (Images in folder + CSV).
    limit: Set this to e.g. 5000 if you want to test quickly without loading all 50k.
    """
    cifar_dir = os.path.join(DATA_DIR, 'cifar-10')
    train_dir = os.path.join(cifar_dir, 'train')
    csv_path = os.path.join(cifar_dir, 'trainLabels.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing labels: {csv_path}")

    # Map text labels to integers
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"Loading CIFAR-10 from {train_dir}...")
    
    images = []
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # Skip header
        
        count = 0
        for row in reader:
            img_id, label_text = row
            img_path = os.path.join(train_dir, f"{img_id}.png")
            
            try:
                # Load image, convert to RGB
                with Image.open(img_path) as img:
                    img_arr = np.array(img)
                    images.append(img_arr)
                    labels.append(class_to_idx[label_text])
            except FileNotFoundError:
                continue
                
            count += 1
            if limit and count >= limit:
                break
                
            if count % 5000 == 0:
                print(f"Loaded {count} images...")

    # Convert to NCHW format: (N, 32, 32, 3) -> (N, 3, 32, 32)
    x_train = np.array(images).astype(np.float32) / 255.0
    x_train = x_train.transpose(0, 3, 1, 2)
    y_train = np.array(labels)
    
    split = int(0.8 * len(x_train))
    return (x_train[:split], y_train[:split]), (x_train[split:], y_train[split:])