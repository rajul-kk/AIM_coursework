import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def run_baseline(X_train, X_test, y_train, y_test):
    """
    Trains a Baseline Random Forest model using all features and default hyperparameters.

    Args:
        X_train, X_test, y_train, y_test: The preprocessed dataset splits.
    
    Returns:
        metrics: A dictionary containing evaluation metrics.
        model: The trained classifier.
    """
    print("\n--- Training Baseline Random Forest ---")
    
    # Initialize the model with default hyperparameters
    # We fix the random state for reproducibility
    rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 1. Train the model and track time
    start_train_time = time.time()
    rf_baseline.fit(X_train, y_train)
    train_time = time.time() - start_train_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    # 2. Test the model and track time
    start_test_time = time.time()
    y_pred = rf_baseline.predict(X_test)
    test_time = time.time() - start_test_time
    print(f"Testing completed in {test_time:.2f} seconds.")
    
    # 3. Calculate Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # We use 'macro' average for multi-class, or 'binary' if simplified
    # 'weighted' is good if classes are imbalanced
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate False Positives from confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        total_fp = int(fp)
    else:
        # Sum of false positives across all classes
        fp_per_class = cm.sum(axis=0) - np.diag(cm)
        total_fp = int(fp_per_class.sum())
    
    print("\n--- Baseline Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"False Positives: {total_fp}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positives': total_fp,
        'train_time': train_time,
        'test_time': test_time,
        'feature_count': X_train.shape[1] # Baseline uses all features
    }
    
    return metrics, rf_baseline
