import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the CICIDS2017 dataset for binary classification.

    Args:
        file_path: Path to the CSV file.

    Returns:
        X_train, X_test, y_train, y_test: Processed splits.
    """
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

    # 1. Clean column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

    # 2. Handle missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 3. Separate features and target
    target_col = 'label'
    if target_col not in df.columns:
         print(f"Error: Target column '{target_col}' not found. Available columns: {df.columns.tolist()[:5]}...")
         return None, None, None, None

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 4. Handle categorical features in X (if any remain after cleaning)
    # The CICIDS2017 features are mostly numerical, but we should handle potential object columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
         X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 5. Encode target label to BINARY (Normal = 0, Attack = 1)
    # Convert exactly 'BENIGN' or 'Normal' to 0, and everything else to 1
    y_binary = y.apply(lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL'] else 1)
    
    # Optional: Display the class distribution
    print("Class distribution:")
    print(pd.Series(y_binary).value_counts())

    # 6. Split data
    # We use a smaller train size if the dataset is massive to speed up optimization later
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42, stratify=y_binary)

    # 7. Scale features
    # Standard scaling is excellent for many ML algorithms and essential for distance-based ones
    # Random Forest doesn't strictly *need* it, but it helps if we later compare with SVM or KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Preprocessing complete. Training shape: {X_train_scaled.shape}, Testing shape: {X_test_scaled.shape}")
    
    # We return the scaler and label encoder in case we need to inverse transform later
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, None
