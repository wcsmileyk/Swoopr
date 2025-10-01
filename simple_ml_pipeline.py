#!/usr/bin/env python3
"""
Simplified ML pipeline focusing on the core ML training
Skip turn segment issues for now and focus on full swoop vs gswoop comparison
"""

import os
import sys
import django
import subprocess
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def parse_gswoop_rotation(output):
    """Parse rotation from gswoop output"""
    rotation_match = re.search(r'degrees of rotation:\s+(\d+)\s+deg\s+\((\w+)-hand\)', output)
    if rotation_match:
        degrees = float(rotation_match.group(1))
        direction = rotation_match.group(2)
        return degrees if direction == 'right' else -degrees
    return None

def extract_basic_features(df, flare_idx, max_gspeed_idx):
    """Extract basic features for ML training"""
    features = {}

    # Basic flight characteristics
    features['flight_duration'] = len(df) * 0.2
    features['turn_duration'] = (max_gspeed_idx - flare_idx) * 0.2

    # Altitude features
    features['entry_altitude'] = df.iloc[flare_idx]['AGL'] / 0.3048  # Convert to feet
    features['max_gspeed_altitude'] = df.iloc[max_gspeed_idx]['AGL'] / 0.3048
    features['altitude_loss'] = (df.iloc[flare_idx]['AGL'] - df.iloc[max_gspeed_idx]['AGL']) / 0.3048

    # Speed features
    features['entry_speed'] = df.iloc[flare_idx]['gspeed'] * 2.23694  # Convert to mph
    features['max_vspeed'] = abs(df.iloc[max_gspeed_idx]['velD']) * 2.23694
    features['max_gspeed'] = df.iloc[max_gspeed_idx]['gspeed'] * 2.23694

    # Heading analysis
    turn_data = df[flare_idx:max_gspeed_idx+1]
    headings = turn_data['heading'].values

    if len(headings) >= 2:
        features['heading_start'] = headings[0]
        features['heading_end'] = headings[-1]

        # Calculate net heading change
        net_change = headings[-1] - headings[0]
        while net_change > 180:
            net_change -= 360
        while net_change < -180:
            net_change += 360
        features['net_heading_change'] = net_change
    else:
        features['heading_start'] = 0
        features['heading_end'] = 0
        features['net_heading_change'] = 0

        # Calculate heading statistics
        heading_changes = []
        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            if abs(diff) <= 90:  # Filter noise
                heading_changes.append(diff)

        if heading_changes:
            features['avg_turn_rate'] = np.mean(np.abs(heading_changes)) / 0.2
            features['max_turn_rate'] = np.max(np.abs(heading_changes)) / 0.2
            features['turn_rate_std'] = np.std(heading_changes)
            features['total_heading_change'] = np.sum(np.abs(heading_changes))

            # Direction consistency
            positive_changes = sum(1 for c in heading_changes if c > 2)
            negative_changes = sum(1 for c in heading_changes if c < -2)
            total_directional = positive_changes + negative_changes
            if total_directional > 0:
                features['direction_consistency'] = abs(positive_changes - negative_changes) / total_directional
            else:
                features['direction_consistency'] = 0
        else:
            features['avg_turn_rate'] = 0
            features['max_turn_rate'] = 0
            features['turn_rate_std'] = 0
            features['total_heading_change'] = 0
            features['direction_consistency'] = 0

    return features

def process_training_file(filepath):
    """Process a single training file"""
    try:
        filename = Path(filepath).name

        # Run gswoop
        result = subprocess.run(['gswoop', '-i', filepath],
                              capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        gswoop_rotation = parse_gswoop_rotation(result.stdout)
        if gswoop_rotation is None:
            return None

        # Analyze with our system
        manager = FlightManager()
        df, metadata = manager.read_flysight_file(filepath)

        landing_idx = manager.get_landing(df)

        try:
            flare_idx = manager.find_flare(df, landing_idx)
        except:
            flare_idx = manager.find_turn_start_fallback(df, landing_idx)

        max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

        # Get our rotation
        our_rotation, intended_turn, confidence, method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

        # Extract features
        features = extract_basic_features(df, flare_idx, max_gspeed_idx)

        return {
            'filename': filename,
            'gswoop_rotation': gswoop_rotation,
            'our_rotation': our_rotation,
            'our_confidence': confidence,
            'our_method': method,
            'our_intended': intended_turn,
            'difference': abs(our_rotation - gswoop_rotation),
            'features': features
        }

    except Exception as e:
        print(f"Error processing {Path(filepath).name}: {e}")
        return None

def generate_simple_dataset(max_files=150):
    """Generate a simple dataset for ML training"""

    print("🤖 SIMPLE ML TRAINING PIPELINE")
    print("=" * 50)

    # Find training files
    training_dir = Path.home() / 'FlySight' / 'Training'
    csv_files = list(training_dir.glob('**/*.csv'))[:max_files]

    print(f"📁 Processing {len(csv_files)} files...")

    # Process files
    training_data = []
    for i, filepath in enumerate(csv_files, 1):
        result = process_training_file(str(filepath))
        if result:
            training_data.append(result)

        if i % 25 == 0:
            print(f"   Processed: {i}/{len(csv_files)}, Success: {len(training_data)}")

    print(f"\n📊 Dataset Complete:")
    print(f"   Total processed: {len(csv_files)}")
    print(f"   Successful: {len(training_data)}")
    print(f"   Success rate: {len(training_data)/len(csv_files)*100:.1f}%")

    if training_data:
        # Analyze differences
        differences = [d['difference'] for d in training_data]
        good_matches = sum(1 for d in differences if d < 50)
        avg_diff = np.mean(differences)

        print(f"   Good matches (<50°): {good_matches}/{len(training_data)} ({good_matches/len(training_data)*100:.1f}%)")
        print(f"   Average difference: {avg_diff:.1f}°")

    return training_data

def train_ml_model(training_data):
    """Train ML model to predict gswoop rotations"""

    print(f"\n🧠 TRAINING ML MODEL")
    print("=" * 40)

    if len(training_data) < 20:
        print("❌ Not enough training data")
        return None

    # Prepare feature matrix and targets
    feature_names = list(training_data[0]['features'].keys())
    X = np.array([[d['features'][f] for f in feature_names] for d in training_data])
    y_gswoop = np.array([d['gswoop_rotation'] for d in training_data])

    print(f"Features: {len(feature_names)}")
    print(f"Training examples: {len(X)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_gswoop, test_size=0.3, random_state=42)

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

    # Compare with our algorithm on test set
    test_indices = X_test.shape[0]
    our_predictions = [training_data[i]['our_rotation'] for i in range(len(training_data))][-test_indices:]
    our_mae = mean_absolute_error(y_test, our_predictions)

    print(f"\n🔄 Comparison:")
    print(f"   Our algorithm MAE: {our_mae:.1f}°")
    print(f"   ML model MAE: {mae:.1f}°")

    improvement = ((our_mae - mae) / our_mae) * 100
    print(f"   Improvement: {improvement:+.1f}%")

    # Feature importance
    print(f"\n🎯 Top Features:")
    feature_importance = sorted(zip(feature_names, model.feature_importances_),
                               key=lambda x: x[1], reverse=True)
    for feature, importance in feature_importance[:5]:
        print(f"   {feature}: {importance:.3f}")

    # Save model
    model_path = Path(__file__).parent / 'rotation_prediction_model.pkl'
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'mae': mae,
        'improvement': improvement
    }, model_path)

    print(f"\n💾 Model saved: {model_path}")

    return model, feature_names, improvement

def main():
    """Main execution"""

    # Generate dataset
    training_data = generate_simple_dataset(max_files=150)

    if len(training_data) > 20:
        # Train model
        result = train_ml_model(training_data)

        if result:
            model, features, improvement = result
            print(f"\n🎉 ML Pipeline Complete!")
            print(f"   Model improvement: {improvement:+.1f}%")

            if improvement > 5:
                print("✅ SIGNIFICANT IMPROVEMENT - Ready for production!")
            elif improvement > 0:
                print("⚡ MARGINAL IMPROVEMENT - Consider more data/features")
            else:
                print("⚠️  NO IMPROVEMENT - Current algorithm sufficient")
    else:
        print("❌ Insufficient training data for ML model")

if __name__ == "__main__":
    main()