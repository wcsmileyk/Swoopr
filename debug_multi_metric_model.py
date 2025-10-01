#!/usr/bin/env python3
"""
Debug multi-metric model structure
"""

import joblib
from pathlib import Path

def debug_model():
    """Debug the multi-metric model structure"""

    model_path = Path('multi_metric_ml_model.joblib')

    print(f"Loading model from {model_path}")
    models = joblib.load(model_path)

    print(f"Model type: {type(models)}")
    print(f"Keys: {list(models.keys())}")

    for key, value in models.items():
        print(f"\n{key}: {type(value)}")
        if key == 'models':
            print(f"  Models available: {list(value.keys())}")
            for model_name, model_data in value.items():
                print(f"    {model_name}: {type(model_data)}")
                if hasattr(model_data, 'keys'):
                    print(f"      Keys: {list(model_data.keys())}")
        elif isinstance(value, list):
            print(f"  List length: {len(value)}")
            print(f"  Sample items: {value[:3]}")

if __name__ == "__main__":
    debug_model()