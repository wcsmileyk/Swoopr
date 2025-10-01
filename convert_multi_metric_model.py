#!/usr/bin/env python3
"""
Convert multi-metric models from pickle to joblib format
"""

import pickle
import joblib
from pathlib import Path

def convert_models():
    """Convert pickle models to joblib format"""

    # Load the pickle file
    pkl_path = Path('multi_metric_models.pkl')
    joblib_path = Path('multi_metric_ml_model.joblib')

    print(f"Loading models from {pkl_path}")

    with open(pkl_path, 'rb') as f:
        models = pickle.load(f)

    print(f"Found {len(models)} models:")
    for metric_name in models.keys():
        print(f"  - {metric_name}")

    # Convert to joblib format
    print(f"Saving models to {joblib_path}")
    joblib.dump(models, joblib_path)

    print("✅ Conversion complete!")

    # Verify the conversion
    loaded = joblib.load(joblib_path)
    print(f"✅ Verification: Loaded {len(loaded)} models from joblib file")

if __name__ == "__main__":
    convert_models()