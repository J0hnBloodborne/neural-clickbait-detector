import os
import matplotlib
# Use Agg backend for non-interactive environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(y_true, y_pred, num_classes=10):
    # Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # Metrics
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
    return cm, precisions, recalls, f1_scores

def print_classification_report(cm, precisions, recalls, f1_scores, class_names=None, file=None):
    if class_names is None:
        class_names = [str(i) for i in range(len(precisions))]

    def out(text):
        print(text)
        if file:
            file.write(text + "\n")
        
    out(f"\n{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    out("-" * 50)
    
    for i, name in enumerate(class_names):
        out(f"{name:<12} | {precisions[i]:.4f}     | {recalls[i]:.4f}     | {f1_scores[i]:.4f}")
        
    out("-" * 50)
    out(f"Average      | {np.mean(precisions):.4f}     | {np.mean(recalls):.4f}     | {np.mean(f1_scores):.4f}")
    
    accuracy = np.trace(cm) / np.sum(cm)
    out(f"Overall Accuracy: {accuracy * 100:.2f}%")

def plot_confusion_matrix(cm, class_names=None, save_path='confusion_matrix.png', title='Confusion Matrix'):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
        
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
                 
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion Matrix heatmap saved to {save_path}")
    plt.close()

def evaluate_model(y_true, y_pred, class_names=None, save_dir='.', prefix='model'):
    # Ensure inputs are numpy arrays (CPU)
    if hasattr(y_true, 'get'): y_true = y_true.get()
    if hasattr(y_pred, 'get'): y_pred = y_pred.get()
    
    # Remove any extra dimensions if they exist (e.g. (N, 1) -> (N,))
    if y_true.ndim > 1: y_true = y_true.flatten()
    if y_pred.ndim > 1: y_pred = y_pred.flatten()

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    num_classes = len(class_names) if class_names else 10
    
    cm, precisions, recalls, f1s = calculate_metrics(y_true, y_pred, num_classes)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    report_path = os.path.join(save_dir, f'{prefix}_report.txt')
    with open(report_path, 'w') as f:
        print_classification_report(cm, precisions, recalls, f1s, class_names, file=f)
    print(f"Classification report saved to {report_path}")
    
    save_path = os.path.join(save_dir, f'{prefix}_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, save_path=save_path)
