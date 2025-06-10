import os
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_dataset():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Define paths - using data/acsincome directory
    data_dir = project_root / "data" / "acsincome"
    y_train_path = data_dir / "y_train.npy"
    
    # Check if files exist
    if not y_train_path.exists():
        print(f"Error: {y_train_path} not found!")
        return
    
    # Load y_train
    y_train = np.load(y_train_path)
    
    # Print basic properties
    print("\n=== Dataset Analysis ===")
    print(f"\n1. y_train Properties:")
    print(f"Shape: {y_train.shape}")
    print(f"Data type: {y_train.dtype}")
    
    # Convert y_train to int if it's float and has a reasonable number of unique values
    if y_train.dtype == np.float64 or y_train.dtype == np.float32:
        unique_values = np.unique(y_train)
        if len(unique_values) < 100:  # Arbitrary threshold for 'reasonable'
            print(f"Converting y_train from {y_train.dtype} to int64 for analysis.")
            y_train = y_train.astype(np.int64)
        else:
            print(f"Warning: y_train is float with {len(unique_values)} unique values. Bincount might not be meaningful.")

    print(f"Unique values: {np.unique(y_train)}")
    try:
        print(f"Value counts: {np.bincount(y_train)}")
    except TypeError as e:
        print(f"Could not compute bincount for y_train: {e}")
        print("This might happen if y_train contains negative or non-integer values after conversion.")

    print(f"Mean: {np.mean(y_train):.4f}")
    print(f"Standard deviation: {np.std(y_train):.4f}")
    
    # Try to load X_train if it exists
    x_train_path = data_dir / "X_train.npy"
    if x_train_path.exists():
        X_train = np.load(x_train_path)
        print(f"\n2. X_train Properties:")
        print(f"Shape: {X_train.shape}")
        print(f"Data type: {X_train.dtype}")
        print(f"Mean: {np.mean(X_train):.4f}")
        print(f"Standard deviation: {np.std(X_train):.4f}")
        print(f"Min value: {np.min(X_train):.4f}")
        print(f"Max value: {np.max(X_train):.4f}")
    
    # Try to load sensitive features if they exist
    sensitive_path = data_dir / "sensitive_features.npy"
    if sensitive_path.exists():
        sensitive_features = np.load(sensitive_path)
        print(f"\n3. Sensitive Features Properties:")
        print(f"Shape: {sensitive_features.shape}")
        print(f"Data type: {sensitive_features.dtype}")
        # Convert sensitive_features to int if it's float and has a reasonable number of unique values
        if sensitive_features.dtype == np.float64 or sensitive_features.dtype == np.float32:
            unique_sensitive_values = np.unique(sensitive_features)
            if len(unique_sensitive_values) < 100:  # Arbitrary threshold
                print(f"Converting sensitive_features from {sensitive_features.dtype} to int64 for analysis.")
                sensitive_features = sensitive_features.astype(np.int64)
            else:
                print(f"Warning: sensitive_features is float with {len(unique_sensitive_values)} unique values. Bincount might not be meaningful.")

        print(f"Unique values: {np.unique(sensitive_features)}")
        try:
            print(f"Value counts: {np.bincount(sensitive_features)}")
        except TypeError as e:
            print(f"Could not compute bincount for sensitive_features: {e}")
            print("This might happen if sensitive_features contains negative or non-integer values after conversion.")
    
    # Print class distribution
    print(f"\n4. Class Distribution:")
    try:
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        for class_idx, count in enumerate(class_counts):
            percentage = (count / total_samples) * 100
            print(f"Class {class_idx}: {count} samples ({percentage:.2f}%)")
    except TypeError as e:
        print(f"Could not compute class distribution for y_train: {e}")

    # Print dataset size information
    print(f"\n5. Dataset Size Information:")
    print(f"Total number of training samples: {len(y_train)}")
    
    # Check for any missing values
    print(f"\n6. Missing Values Check:")
    print(f"Number of NaN values in y_train: {np.isnan(y_train).sum()}")
    if x_train_path.exists():
        print(f"Number of NaN values in X_train: {np.isnan(X_train).sum()}")

if __name__ == "__main__":
    analyze_dataset() 