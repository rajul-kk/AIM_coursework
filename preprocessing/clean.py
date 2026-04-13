import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _normalize_columns(df):
    """Normalize column names to the project-wide convention."""
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df


def _to_binary_label(series):
    """Map BENIGN/NORMAL -> 0 and everything else -> 1."""
    return series.apply(lambda x: 0 if str(x).strip().upper() in ['BENIGN', 'NORMAL'] else 1)


def build_sampled_ml_combined_splits(
    input_dir='data/MachineLearningCVE',
    output_dir='data/combined_ml_15pct',
    sample_frac=0.15,
    sampling_scope='full_concat',
    test_size=0.3,
    random_state=42,
    force_rebuild=False,
):
    """
    Build a sampled-and-combined train/test split from MachineLearningCVE CSV files.

    Sampling scopes:
    - 'full_concat': concatenate all cleaned rows first, then random-sample globally.
    - 'per_file': sample each file independently, then concatenate.

    After sampling, the function performs a stratified train/test split.

    Returns:
        train_path, test_path, stats_path
    """
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError('sample_frac must be in (0, 1].')
    if sampling_scope not in {'full_concat', 'per_file'}:
        raise ValueError("sampling_scope must be 'full_concat' or 'per_file'.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / 'train_sampled.csv'
    test_path = output_path / 'test_sampled.csv'
    stats_path = output_path / 'split_stats.csv'

    if train_path.exists() and test_path.exists() and stats_path.exists() and not force_rebuild:
        try:
            existing_stats = pd.read_csv(stats_path)
            has_meta = {'sampling_scope', 'sample_frac'}.issubset(existing_stats.columns)
            if has_meta and not existing_stats.empty:
                scope_ok = existing_stats['sampling_scope'].astype(str).eq(sampling_scope).all()
                frac_ok = np.isclose(float(existing_stats['sample_frac'].iloc[0]), float(sample_frac))
                if scope_ok and frac_ok:
                    print(f'Using existing sampled split at {output_path}.')
                    return str(train_path), str(test_path), str(stats_path)
                print('Existing split metadata differs; rebuilding sampled split...')
            else:
                print('Existing split metadata missing; rebuilding sampled split...')
        except Exception:
            print('Could not validate existing split metadata; rebuilding sampled split...')

    csv_files = sorted(input_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {input_path}.')

    cleaned_frames = []
    file_stats = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Fail fast if this is a Git LFS pointer rather than real CSV content.
        if len(df.columns) == 1 and str(df.columns[0]).strip().lower().startswith('version https://git-lfs.github.com/spec/v1'):
            raise RuntimeError(
                f'Dataset file is a Git LFS pointer: {csv_file}. Run: git lfs pull origin main'
            )

        df = _normalize_columns(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if 'label' not in df.columns:
            raise RuntimeError(f"Target column 'label' not found in {csv_file}.")

        df['source_file'] = csv_file.name
        cleaned_frames.append(df)

        file_stats.append(
            {
                'source_file': csv_file.name,
                'rows_after_clean': len(df),
            }
        )

    cleaned_combined = pd.concat(cleaned_frames, ignore_index=True)

    if sampling_scope == 'full_concat':
        combined = cleaned_combined.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    else:
        sampled_frames = []
        for idx, frame in enumerate(cleaned_frames):
            sampled_frames.append(frame.sample(frac=sample_frac, random_state=random_state + idx))
        combined = pd.concat(sampled_frames, ignore_index=True)

    # Attach per-source sampled counts and class composition.
    combined_binary = _to_binary_label(combined['label'])
    sampled_source_counts = combined['source_file'].value_counts().to_dict()
    sampled_binary_by_source = (
        pd.DataFrame({'source_file': combined['source_file'], 'label_bin': combined_binary})
        .groupby('source_file')['label_bin']
        .agg(benign_rows=lambda s: int((s == 0).sum()), attack_rows=lambda s: int((s == 1).sum()))
        .to_dict('index')
    )

    for row in file_stats:
        src = row['source_file']
        row['rows_sampled'] = int(sampled_source_counts.get(src, 0))
        row['benign_rows'] = int(sampled_binary_by_source.get(src, {}).get('benign_rows', 0))
        row['attack_rows'] = int(sampled_binary_by_source.get(src, {}).get('attack_rows', 0))

    y_binary = _to_binary_label(combined['label'])

    train_df, test_df = train_test_split(
        combined,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binary,
    )

    # Save split files without index for notebook/script reuse.
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    train_binary = _to_binary_label(train_df['label'])
    test_binary = _to_binary_label(test_df['label'])

    stats_rows = file_stats + [
        {
            'source_file': '__combined_train__',
            'rows_after_clean': len(train_df),
            'rows_sampled': len(train_df),
            'benign_rows': int((train_binary == 0).sum()),
            'attack_rows': int((train_binary == 1).sum()),
            'sampling_scope': sampling_scope,
            'sample_frac': sample_frac,
        },
        {
            'source_file': '__combined_test__',
            'rows_after_clean': len(test_df),
            'rows_sampled': len(test_df),
            'benign_rows': int((test_binary == 0).sum()),
            'attack_rows': int((test_binary == 1).sum()),
            'sampling_scope': sampling_scope,
            'sample_frac': sample_frac,
        },
        {
            'source_file': '__combined_total__',
            'rows_after_clean': len(combined),
            'rows_sampled': len(combined),
            'benign_rows': int((combined_binary == 0).sum()),
            'attack_rows': int((combined_binary == 1).sum()),
            'sampling_scope': sampling_scope,
            'sample_frac': sample_frac,
        },
    ]

    for row in file_stats:
        row['sampling_scope'] = sampling_scope
        row['sample_frac'] = sample_frac

    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    print(f'Sampled combined split saved to {output_path}.')
    print(f'Train rows: {len(train_df)} | Test rows: {len(test_df)} | Sample fraction: {sample_frac}')
    return str(train_path), str(test_path), str(stats_path)


def load_pre_split_data(train_path, test_path):
    """
    Load pre-split CSVs (already train/test) and apply the standard project preprocessing.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, None
    """
    empty_result = (None, None, None, None, None, None)
    print(f'Loading pre-split data from {train_path} and {test_path}...')
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f'Error loading pre-split data: {e}')
        return empty_result

    train_df = _normalize_columns(train_df)
    test_df = _normalize_columns(test_df)

    for df in (train_df, test_df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    if 'label' not in train_df.columns or 'label' not in test_df.columns:
        print("Error: Target column 'label' not found in split files.")
        return empty_result

    y_train = _to_binary_label(train_df['label'])
    y_test = _to_binary_label(test_df['label'])

    # Keep source_file as metadata only and out of model features.
    drop_cols = ['label']
    if 'source_file' in train_df.columns and 'source_file' in test_df.columns:
        drop_cols.append('source_file')

    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    # Encode any remaining categorical columns with a shared train+test vocabulary.
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        all_vals = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(all_vals)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f'Pre-split preprocessing complete. Training shape: {X_train_scaled.shape}, Testing shape: {X_test_scaled.shape}')
    print('Class distribution (train):')
    print(pd.Series(y_train).value_counts())
    print('Class distribution (test):')
    print(pd.Series(y_test).value_counts())

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, None

def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the CICIDS2017 dataset for binary classification.

    Args:
        file_path: Path to the CSV file.

    Returns:
        X_train, X_test, y_train, y_test: Processed splits.
    """
    empty_result = (None, None, None, None, None, None)
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return empty_result

    # If a CSV is still an LFS pointer file, fail fast with an actionable message.
    if len(df.columns) == 1 and str(df.columns[0]).strip().lower().startswith('version https://git-lfs.github.com/spec/v1'):
        print("Error: This dataset file is a Git LFS pointer, not the real CSV.")
        print("Run: git lfs pull origin main")
        return empty_result

    # 1. Clean column names
    df = _normalize_columns(df)

    # 2. Handle missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 3. Separate features and target
    target_col = 'label'
    if target_col not in df.columns:
           print(f"Error: Target column '{target_col}' not found. Available columns: {df.columns.tolist()[:5]}...")
           return empty_result

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # 4. Handle categorical features in X (if any remain after cleaning)
    # The CICIDS2017 features are mostly numerical, but we should handle potential object columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
         X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 5. Encode target label to BINARY (Normal = 0, Attack = 1)
    # Convert exactly 'BENIGN' or 'Normal' to 0, and everything else to 1
    y_binary = _to_binary_label(y)
    
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
