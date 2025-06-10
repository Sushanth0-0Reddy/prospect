import pickle
import argparse
import pandas as pd
import torch
import numpy as np
import sys

# Helper to get a string representation for printing large objects
def get_repr(obj, max_len=100):
    s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "... (truncated)"
    return s

def main():
    parser = argparse.ArgumentParser(description="Inspect a seed_X.p file.")
    parser.add_argument("file_path", type=str, help="Path to the seed_X.p file to inspect.")
    args = parser.parse_args()

    print(f"Inspecting file: {args.file_path}")

    try:
        with open(args.file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)

    print(f"\nLoaded data type: {type(data)}")

    if isinstance(data, dict):
        print("\nData dictionary keys:")
        for key in data.keys():
            print(f"  - {key} (type: {type(data[key])})")

        if "metrics" in data and isinstance(data["metrics"], pd.DataFrame):
            print("\n--- Metrics DataFrame --- DRAFT")
            metrics_df = data["metrics"]
            print(f"Metrics DataFrame columns: {metrics_df.columns.tolist()}")
            print(f"Metrics DataFrame shape: {metrics_df.shape}")

            if not metrics_df.empty:
                print(f"\nFirst few rows of metrics DataFrame (selected columns):")
                cols_to_show = ['epoch', 'train_loss', 'val_loss', 'elapsed', 'iterations']
                existing_cols_to_show = [col for col in cols_to_show if col in metrics_df.columns]
                
                # For the 'iterations' column, show type and shape of the first element
                if 'iterations' in existing_cols_to_show:
                    first_iter_val = metrics_df['iterations'].iloc[0]
                    print(f"  Type of first element in 'iterations' column: {type(first_iter_val)}")
                    if hasattr(first_iter_val, 'shape'):
                        print(f"  Shape of first element in 'iterations' column: {first_iter_val.shape}")
                    
                    # Create a version of the dataframe for printing that summarizes the 'iterations' column
                    metrics_print_df = metrics_df[existing_cols_to_show].copy()
                    if 'iterations' in metrics_print_df.columns:
                         metrics_print_df['iterations_repr'] = metrics_print_df['iterations'].apply(lambda x: f"{type(x).__name__} shape {x.shape if hasattr(x, 'shape') else '(no shape)'}")
                         # metrics_print_df['iterations_repr'] = metrics_print_df['iterations'].apply(get_repr) # Alternative: full repr truncated
                    print(metrics_print_df.head())
                else:
                    print(metrics_df[existing_cols_to_show].head())

            else:
                print("Metrics DataFrame is empty.")
        
        if "weights" in data:
            print("\n--- Final Weights ('weights' key) ---")
            final_weights = data["weights"]
            print(f"Type of final weights: {type(final_weights)}")
            if hasattr(final_weights, 'shape'):
                print(f"Shape of final weights: {final_weights.shape}")
            else:
                print(f"Final weights: {get_repr(final_weights)}")
                
    elif isinstance(data, int) and data == -1: # FAIL_CODE
        print("File contains FAIL_CODE (-1), indicating a diverged or failed run.")
    else:
        print("Loaded data is not a dictionary. Printing its representation (may be large):")
        print(get_repr(data, max_len=500))

if __name__ == "__main__":
    main() 