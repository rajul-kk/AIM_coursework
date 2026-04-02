import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def decode_particle(particle, num_features):
    """
    Decodes the continuous particle array into feature selection mask and hyperparameters.
    """
    # 1. Feature selection (first num_features elements)
    feature_mask = particle[:num_features] > 0.5
    
    # Ensure at least one feature is selected
    if not np.any(feature_mask):
        feature_mask[np.random.randint(0, num_features)] = True
        
    # 2. Hyperparameters (remaining 3 elements)
    n_estimators = int(10 + particle[num_features] * (200 - 10))
    max_depth = int(5 + particle[num_features + 1] * (50 - 5))
    min_samples_split = int(2 + particle[num_features + 2] * (20 - 2))
    
    return feature_mask, n_estimators, max_depth, min_samples_split

def evaluate_solution(particle, X_train, y_train, X_val, y_val):
    """
    Evaluates a single particle's fitness (cost to be minimized).
    """
    num_features = X_train.shape[1]
    feature_mask, n_est, m_depth, min_split = decode_particle(particle, num_features)
    
    X_train_filtered = X_train[:, feature_mask]
    X_val_filtered = X_val[:, feature_mask]
    
    rf = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=m_depth,
        min_samples_split=min_split,
        random_state=42,
        n_jobs=-1
    )
    
    start_fit = time.time()
    rf.fit(X_train_filtered, y_train)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    y_pred = rf.predict(X_val_filtered)
    pred_time = time.time() - start_pred
    
    acc = accuracy_score(y_val, y_pred)
    selected_count = np.sum(feature_mask)

    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        # Multi-class fallback: total false positives over all predictions.
        total_fp = (cm.sum(axis=0) - np.diag(cm)).sum()
        fp_rate = total_fp / cm.sum() if cm.sum() > 0 else 0.0

    feature_reduction = 1.0 - (selected_count / num_features)

    # Keep runtime penalty bounded to [0, 1] for stable optimization pressure.
    total_runtime = fit_time + pred_time
    runtime_penalty = min(total_runtime, 2.0) / 2.0

    # Maximize score by rewarding accuracy/reduction/low-FPR/low-runtime.
    score = (
        0.55 * acc +
        0.20 * feature_reduction +
        0.15 * (1.0 - fp_rate) +
        0.10 * (1.0 - runtime_penalty)
    )

    # Cost minimized by GA/PSO/GWO.
    cost = -1.0 * score
    return cost

def evaluate_final_model(best_particle, X_train, y_train, X_test, y_test):
    """
    Trains and returns the final model and its metrics using the best particle found.
    Calculates False Positives along with other metrics.
    """
    print("\n--- Training Optimized Random Forest ---")
    num_features = X_train.shape[1]
    feature_mask, n_est, m_depth, min_split = decode_particle(best_particle, num_features)
    
    print(f"Optimized Hyperparameters -> n_estimators: {n_est}, max_depth: {m_depth}, min_samples_split: {min_split}")
    print(f"Features Selected: {np.sum(feature_mask)} / {num_features}")
    
    X_train_filtered = X_train[:, feature_mask]
    X_test_filtered = X_test[:, feature_mask]
    
    rf_opt = RandomForestClassifier(
        n_estimators=n_est,
        max_depth=m_depth,
        min_samples_split=min_split,
        random_state=42,
        n_jobs=-1
    )
    
    start_train_time = time.time()
    rf_opt.fit(X_train_filtered, y_train)
    train_time = time.time() - start_train_time
    print(f"Training completed in {train_time:.2f} seconds.")
    
    start_test_time = time.time()
    y_pred = rf_opt.predict(X_test_filtered)
    test_time = time.time() - start_test_time
    print(f"Testing completed in {test_time:.2f} seconds.")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        total_fp = int(fp)
    else:
        fp_per_class = cm.sum(axis=0) - np.diag(cm)
        total_fp = int(fp_per_class.sum())
    
    print("\n--- Optimized Results ---")
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
        'feature_count': np.sum(feature_mask)
    }
    
    return metrics, rf_opt, feature_mask
