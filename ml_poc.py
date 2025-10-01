#!/usr/bin/env python3
"""
Machine Learning Proof of Concept for Swoop Analysis
Compare gswoop ground truth vs our algorithm, train ML model
"""

import os
import sys
import django
import subprocess
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def parse_gswoop_output(output):
    """Parse gswoop command output to extract key metrics"""
    data = {}

    # Extract rotation degrees
    rotation_match = re.search(r'degrees of rotation:\s+(\d+)\s+deg\s+\((\w+)-hand\)', output)
    if rotation_match:
        degrees = float(rotation_match.group(1))
        direction = rotation_match.group(2)
        data['gswoop_rotation'] = -degrees if direction == 'left' else degrees

    # Extract other key metrics
    patterns = {
        'initiated_turn_alt': r'initiated turn:\s+(\d+)\s+ft AGL',
        'max_vspeed': r'max vertical speed:.*\((\d+\.\d+)\s+mph\)',
        'max_hspeed': r'max horizontal speed:.*\((\d+\.\d+)\s+mph\)',
        'turn_time': r'time to execute turn:\s+(\d+\.\d+)\s+sec',
        'entry_speed': r'entry gate speed:\s+(\d+\.\d+)\s+mph',
        'distance_stop': r'distance to stop:\s+(\d+)\s+ft'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            data[key] = float(match.group(1))

    return data

def extract_features(df, flare_idx, max_gspeed_idx, landing_idx):
    """Extract features from GPS data for ML training"""
    features = {}

    if flare_idx is None or max_gspeed_idx is None or landing_idx is None:
        return None

    # Turn segment data
    turn_data = df[flare_idx:max_gspeed_idx+1]
    full_swoop = df[flare_idx:landing_idx+1]

    if len(turn_data) < 3:
        return None

    # Basic flight characteristics
    features['flight_duration'] = len(df) * 0.2  # 5Hz data
    features['turn_duration'] = len(turn_data) * 0.2
    features['swoop_duration'] = len(full_swoop) * 0.2

    # Altitude features
    features['entry_altitude'] = df.iloc[flare_idx]['AGL']
    features['max_vspeed_altitude'] = df.iloc[max_gspeed_idx]['AGL']
    features['landing_altitude'] = df.iloc[landing_idx]['AGL']
    features['altitude_loss'] = features['entry_altitude'] - features['landing_altitude']

    # Speed features
    features['entry_speed'] = df.iloc[flare_idx]['gspeed'] * 2.23694  # Convert to mph
    features['max_vspeed'] = abs(df.iloc[max_gspeed_idx]['velD']) * 2.23694
    features['max_gspeed'] = df.iloc[max_gspeed_idx]['gspeed'] * 2.23694

    # Heading analysis
    headings = turn_data['heading'].values
    features['heading_start'] = headings[0]
    features['heading_end'] = headings[-1]

    # Calculate heading changes
    heading_changes = []
    for i in range(1, len(headings)):
        diff = headings[i] - headings[i-1]
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        heading_changes.append(diff)

    if heading_changes:
        features['avg_turn_rate'] = np.mean(np.abs(heading_changes)) / 0.2  # degrees per second
        features['max_turn_rate'] = np.max(np.abs(heading_changes)) / 0.2
        features['turn_rate_std'] = np.std(heading_changes)
        features['total_heading_change'] = np.sum(np.abs(heading_changes))

        # Direction consistency
        positive_changes = sum(1 for c in heading_changes if c > 0)
        negative_changes = sum(1 for c in heading_changes if c < 0)
        features['direction_consistency'] = abs(positive_changes - negative_changes) / len(heading_changes)

    # GPS accuracy during turn
    if 'hAcc' in turn_data.columns:
        features['avg_horizontal_accuracy'] = turn_data['hAcc'].mean()
        features['max_horizontal_accuracy'] = turn_data['hAcc'].max()

    return features

def generate_training_data():
    """Generate training dataset from sample files"""

    # Read sample file list
    with open('/tmp/sample_files.txt', 'r') as f:
        sample_files = [line.strip() for line in f.readlines()]

    print(f"🚀 Processing {len(sample_files)} files for ML training dataset...")
    print("=" * 60)

    training_data = []
    manager = FlightManager()

    for i, filepath in enumerate(sample_files, 1):
        try:
            print(f"📁 {i:2d}/50: {Path(filepath).name}")

            # Run gswoop to get ground truth
            result = subprocess.run(['gswoop', '-i', filepath],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                print(f"   ❌ gswoop failed: {result.stderr}")
                continue

            gswoop_data = parse_gswoop_output(result.stdout)
            if 'gswoop_rotation' not in gswoop_data:
                print(f"   ⚠️  Could not parse rotation from gswoop output")
                continue

            # Analyze with our algorithm
            df, metadata = manager.read_flysight_file(filepath)

            # Find key points
            landing_idx = manager.get_landing(df)
            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Get our algorithm's results
            our_rotation, intended_turn, confidence, method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

            # Extract features
            features = extract_features(df, flare_idx, max_gspeed_idx, landing_idx)
            if features is None:
                print(f"   ⚠️  Could not extract features")
                continue

            # Combine all data
            row = {
                'filename': Path(filepath).name,
                'gswoop_rotation': gswoop_data['gswoop_rotation'],
                'our_rotation': our_rotation,
                'our_intended': intended_turn,
                'our_confidence': confidence,
                'our_method': method,
                **features,
                **{k: v for k, v in gswoop_data.items() if k != 'gswoop_rotation'}
            }

            training_data.append(row)

            # Show comparison
            diff = abs(our_rotation - gswoop_data['gswoop_rotation'])
            status = "✅" if diff < 20 else "⚠️" if diff < 50 else "❌"
            print(f"   {status} Our: {our_rotation:.1f}° | gswoop: {gswoop_data['gswoop_rotation']:.1f}° | diff: {diff:.1f}°")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    if not training_data:
        print("❌ No training data generated!")
        return None

    # Create DataFrame and save
    df_train = pd.DataFrame(training_data)
    output_file = '/home/smiley/PycharmProjects/Swoopr/ml_training_data.csv'
    df_train.to_csv(output_file, index=False)

    print(f"\n📊 Training dataset created: {len(df_train)} samples")
    print(f"💾 Saved to: {output_file}")

    # Quick analysis
    rotation_diff = np.abs(df_train['our_rotation'] - df_train['gswoop_rotation'])
    print(f"\n📈 Algorithm Comparison:")
    print(f"   Mean rotation error: {rotation_diff.mean():.1f}°")
    print(f"   Median rotation error: {rotation_diff.median():.1f}°")
    print(f"   Within 20°: {(rotation_diff <= 20).sum()}/{len(df_train)} ({(rotation_diff <= 20).mean()*100:.1f}%)")
    print(f"   Within 50°: {(rotation_diff <= 50).sum()}/{len(df_train)} ({(rotation_diff <= 50).mean()*100:.1f}%)")

    return df_train

def train_ml_model(df_train):
    """Train a simple ML model to predict rotation"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib

    print(f"\n🧠 Training ML Model...")
    print("=" * 40)

    # Select features for training
    feature_cols = [col for col in df_train.columns
                   if col not in ['filename', 'gswoop_rotation', 'our_rotation', 'our_intended',
                                 'our_confidence', 'our_method'] and not col.startswith('gswoop_')]

    X = df_train[feature_cols].fillna(0)
    y = df_train['gswoop_rotation']  # Ground truth from gswoop

    print(f"Features used: {len(feature_cols)}")
    print(f"Training samples: {len(X)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n📊 ML Model Results:")
    print(f"   Mean Absolute Error: {mae:.1f}°")
    print(f"   Root Mean Square Error: {rmse:.1f}°")
    print(f"   Within 20°: {(np.abs(y_test - y_pred) <= 20).mean()*100:.1f}%")
    print(f"   Within 50°: {(np.abs(y_test - y_pred) <= 50).mean()*100:.1f}%")

    # Compare with our algorithm on test set
    test_indices = X_test.index
    our_pred = df_train.loc[test_indices, 'our_rotation']
    our_mae = mean_absolute_error(y_test, our_pred)

    print(f"\n🔄 Comparison vs Our Algorithm:")
    print(f"   Our algorithm MAE: {our_mae:.1f}°")
    print(f"   ML model MAE: {mae:.1f}°")
    improvement = ((our_mae - mae) / our_mae) * 100
    print(f"   Improvement: {improvement:+.1f}%")

    # Feature importance
    print(f"\n🎯 Top 5 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    # Save model
    model_path = '/home/smiley/PycharmProjects/Swoopr/rotation_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

    return model, feature_cols, improvement

if __name__ == "__main__":
    # Generate training data
    df_train = generate_training_data()

    if df_train is not None and len(df_train) > 10:
        # Train ML model
        model, features, improvement = train_ml_model(df_train)

        print(f"\n🎉 Proof of Concept Complete!")
        print(f"   Training samples: {len(df_train)}")
        print(f"   Algorithm improvement: {improvement:+.1f}%")

        if improvement > 10:
            print(f"   ✅ RECOMMENDATION: Scale to full dataset (637 files)")
        elif improvement > 0:
            print(f"   ⚠️  MARGINAL: Consider more features or larger dataset")
        else:
            print(f"   ❌ POOR: Current algorithm may be sufficient")
    else:
        print(f"❌ Insufficient training data generated")