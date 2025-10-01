#!/usr/bin/env python3
"""
Multi-Metric ML Pipeline - Expand ML beyond rotation to all gswoop metrics
Uses gswoop as ground truth for comprehensive swoop metric prediction
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
from sklearn.preprocessing import StandardScaler
import joblib
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

@dataclass
class GswoopMetrics:
    """Container for all gswoop metrics"""
    # Key performance metrics
    rotation_degrees: float
    rotation_direction: str  # 'left' or 'right'
    max_vertical_speed_mph: float
    max_total_speed_mph: float
    max_horizontal_speed_mph: float
    entry_gate_speed_mph: float

    # Timing metrics
    time_to_execute_turn: float
    time_during_rollout: float
    time_aloft_during_swoop: float

    # Distance/position metrics
    distance_to_stop: Optional[float]
    touchdown_estimate: Optional[float]
    touchdown_speed_mph: Optional[float]

    # Altitude metrics (converted to feet)
    airplane_exit_altitude: float
    turn_initiation_altitude: float
    max_vspeed_altitude: float
    rollout_start_altitude: float
    rollout_finish_altitude: float
    max_total_speed_altitude: float
    max_hspeed_altitude: float

    # Position offsets (relative to gate)
    turn_init_back: float
    turn_init_offset: float
    max_vspeed_back: float
    max_vspeed_offset: float
    rollout_start_back: float
    rollout_start_offset: float
    max_total_back: float
    max_total_offset: float
    max_hspeed_back: float
    max_hspeed_offset: float

def parse_gswoop_output(output: str) -> Optional[GswoopMetrics]:
    """Parse comprehensive gswoop output into structured metrics"""

    try:
        lines = output.strip().split('\n')
        metrics = {}

        # Parse altitude and position lines
        for line in lines:
            if 'exited airplane:' in line:
                # "exited airplane:      5877 ft AGL"
                match = re.search(r'exited airplane:\s+(\d+) ft AGL', line)
                if match:
                    metrics['airplane_exit_altitude'] = float(match.group(1))

            elif 'initiated turn:' in line:
                # "initiated turn:        978 ft AGL,  625 ft back, -412 ft offset"
                match = re.search(r'initiated turn:\s+(\d+) ft AGL,\s+(\d+) ft back,\s+(-?\d+) ft offset', line)
                if match:
                    metrics['turn_initiation_altitude'] = float(match.group(1))
                    metrics['turn_init_back'] = float(match.group(2))
                    metrics['turn_init_offset'] = float(match.group(3))

            elif 'max vertical speed:' in line:
                # "max vertical speed:    473 ft AGL,  464 ft back,  -12 ft offset (76.6 mph)"
                match = re.search(r'max vertical speed:\s+(\d+) ft AGL,\s+(\d+) ft back,\s+(-?\d+) ft offset \((\d+\.?\d*) mph\)', line)
                if match:
                    metrics['max_vspeed_altitude'] = float(match.group(1))
                    metrics['max_vspeed_back'] = float(match.group(2))
                    metrics['max_vspeed_offset'] = float(match.group(3))
                    metrics['max_vertical_speed_mph'] = float(match.group(4))

            elif 'started rollout:' in line:
                # "started rollout:       318 ft AGL,  457 ft back,   22 ft offset (75.4 mph)"
                match = re.search(r'started rollout:\s+(\d+) ft AGL,\s+(\d+) ft back,\s+(-?\d+) ft offset', line)
                if match:
                    metrics['rollout_start_altitude'] = float(match.group(1))
                    metrics['rollout_start_back'] = float(match.group(2))
                    metrics['rollout_start_offset'] = float(match.group(3))

            elif 'finished rollout:' in line:
                # "finished rollout:       18 ft AGL,    0 ft back,    0 ft offset"
                match = re.search(r'finished rollout:\s+(\d+) ft AGL', line)
                if match:
                    metrics['rollout_finish_altitude'] = float(match.group(1))

            elif 'max total speed:' in line:
                # "max total speed:       231 ft AGL,  418 ft back,   28 ft offset (82.4 mph)"
                match = re.search(r'max total speed:\s+(\d+) ft AGL,\s+(\d+) ft back,\s+(-?\d+) ft offset \((\d+\.?\d*) mph\)', line)
                if match:
                    metrics['max_total_speed_altitude'] = float(match.group(1))
                    metrics['max_total_back'] = float(match.group(2))
                    metrics['max_total_offset'] = float(match.group(3))
                    metrics['max_total_speed_mph'] = float(match.group(4))

            elif 'max horizontal speed:' in line:
                # "max horizontal speed:   57 ft AGL,  209 ft back,   -2 ft offset (68.7 mph)"
                match = re.search(r'max horizontal speed:\s+(\d+) ft AGL,\s+(\d+) ft back,\s+(-?\d+) ft offset \((\d+\.?\d*) mph\)', line)
                if match:
                    metrics['max_hspeed_altitude'] = float(match.group(1))
                    metrics['max_hspeed_back'] = float(match.group(2))
                    metrics['max_hspeed_offset'] = float(match.group(3))
                    metrics['max_horizontal_speed_mph'] = float(match.group(4))

        # Parse rotation
        rotation_match = re.search(r'degrees of rotation:\s+(\d+) deg \((\w+)-hand\)', output)
        if rotation_match:
            degrees = float(rotation_match.group(1))
            direction = rotation_match.group(2)
            metrics['rotation_degrees'] = degrees if direction == 'right' else -degrees
            metrics['rotation_direction'] = direction

        # Parse timing metrics
        time_matches = [
            (r'time to execute turn:\s+(\d+\.?\d*) sec', 'time_to_execute_turn'),
            (r'time during rollout:\s+(\d+\.?\d*) sec', 'time_during_rollout'),
            (r'time aloft during swoop:\s+(\d+\.?\d*) sec', 'time_aloft_during_swoop')
        ]

        for pattern, key in time_matches:
            match = re.search(pattern, output)
            if match:
                metrics[key] = float(match.group(1))

        # Parse speed and distance metrics
        entry_speed_match = re.search(r'entry gate speed:\s+(\d+\.?\d*) mph', output)
        if entry_speed_match:
            metrics['entry_gate_speed_mph'] = float(entry_speed_match.group(1))

        distance_match = re.search(r'distance to stop:\s+(\d+) ft', output)
        if distance_match:
            metrics['distance_to_stop'] = float(distance_match.group(1))

        touchdown_match = re.search(r'touchdown estimate:\s+(\d+) ft \((\d+\.?\d*) mph\)', output)
        if touchdown_match:
            metrics['touchdown_estimate'] = float(touchdown_match.group(1))
            metrics['touchdown_speed_mph'] = float(touchdown_match.group(2))

        # Validate we have the core metrics
        required_fields = ['rotation_degrees', 'max_vertical_speed_mph', 'entry_gate_speed_mph', 'time_to_execute_turn']
        if all(field in metrics for field in required_fields):
            return GswoopMetrics(**{field: metrics.get(field, 0.0) for field in GswoopMetrics.__annotations__})

        return None

    except Exception as e:
        print(f"Error parsing gswoop output: {e}")
        return None

def extract_comprehensive_features(df, flare_idx, max_gspeed_idx) -> Dict[str, float]:
    """Extract comprehensive features for multi-metric ML prediction"""
    features = {}

    # Basic flight characteristics
    features['flight_duration'] = len(df) * 0.2
    features['turn_duration'] = (max_gspeed_idx - flare_idx) * 0.2
    features['data_points'] = len(df)

    # Altitude features (all in feet)
    features['entry_altitude'] = df.iloc[flare_idx]['AGL'] / 0.3048
    features['max_gspeed_altitude'] = df.iloc[max_gspeed_idx]['AGL'] / 0.3048
    features['altitude_loss'] = (df.iloc[flare_idx]['AGL'] - df.iloc[max_gspeed_idx]['AGL']) / 0.3048
    features['landing_altitude'] = df.iloc[-1]['AGL'] / 0.3048

    # Altitude statistics during turn
    turn_data = df[flare_idx:max_gspeed_idx+1]
    turn_altitudes = turn_data['AGL'].values / 0.3048  # Convert to feet
    features['turn_alt_mean'] = np.mean(turn_altitudes)
    features['turn_alt_std'] = np.std(turn_altitudes)
    features['turn_alt_range'] = np.max(turn_altitudes) - np.min(turn_altitudes)

    # Speed features (all in mph)
    features['entry_speed'] = df.iloc[flare_idx]['gspeed'] * 2.23694
    features['max_vspeed'] = abs(df.iloc[max_gspeed_idx]['velD']) * 2.23694
    features['max_gspeed'] = df.iloc[max_gspeed_idx]['gspeed'] * 2.23694
    features['landing_speed'] = df.iloc[-1]['gspeed'] * 2.23694

    # Speed statistics during turn
    turn_gspeeds = turn_data['gspeed'].values * 2.23694  # Convert to mph
    turn_vspeeds = np.abs(turn_data['velD'].values) * 2.23694
    features['turn_gspeed_mean'] = np.mean(turn_gspeeds)
    features['turn_gspeed_std'] = np.std(turn_gspeeds)
    features['turn_vspeed_mean'] = np.mean(turn_vspeeds)
    features['turn_vspeed_std'] = np.std(turn_vspeeds)

    # Find max total speed
    turn_total_speeds = np.sqrt(turn_data['velN']**2 + turn_data['velE']**2 + turn_data['velD']**2) * 2.23694
    features['max_total_speed'] = np.max(turn_total_speeds)

    # Velocity component analysis
    features['max_vel_north'] = np.max(np.abs(turn_data['velN'])) * 2.23694
    features['max_vel_east'] = np.max(np.abs(turn_data['velE'])) * 2.23694
    features['max_vel_down'] = np.max(np.abs(turn_data['velD'])) * 2.23694

    # Heading analysis
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

        # Heading rate analysis
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
    else:
        # Default values for insufficient heading data
        for key in ['heading_start', 'heading_end', 'net_heading_change', 'avg_turn_rate',
                   'max_turn_rate', 'turn_rate_std', 'total_heading_change', 'direction_consistency']:
            features[key] = 0

    # Acceleration features
    if len(turn_data) > 2:
        # Calculate accelerations (rough approximation)
        vel_n = turn_data['velN'].values
        vel_e = turn_data['velE'].values
        vel_d = turn_data['velD'].values

        acc_n = np.diff(vel_n) / 0.2  # m/s²
        acc_e = np.diff(vel_e) / 0.2
        acc_d = np.diff(vel_d) / 0.2

        features['max_accel_north'] = np.max(np.abs(acc_n)) * 2.23694 if len(acc_n) > 0 else 0
        features['max_accel_east'] = np.max(np.abs(acc_e)) * 2.23694 if len(acc_e) > 0 else 0
        features['max_accel_down'] = np.max(np.abs(acc_d)) * 2.23694 if len(acc_d) > 0 else 0

        total_accel = np.sqrt(acc_n**2 + acc_e**2 + acc_d**2)
        features['max_total_accel'] = np.max(total_accel) * 2.23694 if len(total_accel) > 0 else 0
    else:
        for key in ['max_accel_north', 'max_accel_east', 'max_accel_down', 'max_total_accel']:
            features[key] = 0

    # Flight path geometry
    if len(turn_data) > 1:
        # Calculate approximate path length during turn
        coords = turn_data[['velN', 'velE', 'velD']].values
        path_segments = np.diff(coords, axis=0)
        path_distances = np.sqrt(np.sum(path_segments**2, axis=1)) * 0.2  # Distance per segment
        features['turn_path_length'] = np.sum(path_distances) * 3.28084  # Convert to feet
        features['turn_path_efficiency'] = features['altitude_loss'] / features['turn_path_length'] if features['turn_path_length'] > 0 else 0
    else:
        features['turn_path_length'] = 0
        features['turn_path_efficiency'] = 0

    return features

def process_training_file_comprehensive(filepath):
    """Process a training file for comprehensive multi-metric learning"""
    try:
        filename = Path(filepath).name

        # Run gswoop to get ground truth metrics
        result = subprocess.run(['gswoop', '-i', filepath],
                              capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return None

        gswoop_metrics = parse_gswoop_output(result.stdout)
        if gswoop_metrics is None:
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

        # Extract comprehensive features
        features = extract_comprehensive_features(df, flare_idx, max_gspeed_idx)

        return {
            'filename': filename,
            'gswoop_metrics': gswoop_metrics,
            'features': features,
            'indices': {
                'flare_idx': flare_idx,
                'max_gspeed_idx': max_gspeed_idx,
                'landing_idx': landing_idx,
                'max_vspeed_idx': max_vspeed_idx
            }
        }

    except Exception as e:
        print(f"Error processing {Path(filepath).name}: {e}")
        return None

def train_multi_metric_models(training_data, metrics_to_train=None):
    """Train separate ML models for different metrics"""

    if metrics_to_train is None:
        # Define which metrics to train models for
        # Skip vertical/horizontal speeds - we calculate those well already
        metrics_to_train = [
            'rotation_degrees',            # Primary target - already proven valuable
            'time_to_execute_turn',        # Complex timing metric
            'time_during_rollout',         # Rollout detection timing
            'time_aloft_during_swoop',     # Overall swoop timing
            'distance_to_stop',            # Distance prediction
            'touchdown_estimate',          # Landing distance prediction
            'touchdown_speed_mph',         # Landing speed prediction
            'entry_gate_speed_mph',        # Gate entry analysis
            # Position metrics - complex spatial calculations
            'turn_init_back',
            'turn_init_offset',
            'max_total_back',
            'max_total_offset'
        ]

    print(f"\n🧠 TRAINING MULTI-METRIC ML MODELS")
    print("=" * 50)
    print(f"Training {len(metrics_to_train)} different models")
    print(f"Training examples: {len(training_data)}")

    if len(training_data) < 20:
        print("❌ Not enough training data")
        return None

    # Prepare feature matrix
    feature_names = list(training_data[0]['features'].keys())
    X = np.array([[d['features'][f] for f in feature_names] for d in training_data])

    # Train models for each metric
    models = {}
    results = {}

    for metric in metrics_to_train:
        print(f"\n📊 Training model for: {metric}")

        # Extract target values
        y = []
        for d in training_data:
            value = getattr(d['gswoop_metrics'], metric, None)
            if value is not None:
                y.append(float(value))
            else:
                y.append(0.0)  # Default value for missing data

        y = np.array(y)

        # Skip if all values are zero or very similar
        if np.std(y) < 0.1:
            print(f"   ⚠️  Skipping {metric} - insufficient variance")
            continue

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Scale features for this model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Calculate improvement (assuming our algorithm would be very basic)
        baseline_mae = np.std(y_test)  # Simple baseline
        improvement = ((baseline_mae - mae) / baseline_mae) * 100 if baseline_mae > 0 else 0

        print(f"   MAE: {mae:.2f}, RMSE: {rmse:.2f}, Improvement: {improvement:+.1f}%")

        # Store model and results
        models[metric] = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'mae': mae,
            'improvement': improvement,
            'target_stats': {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y)
            }
        }

        results[metric] = {
            'mae': mae,
            'rmse': rmse,
            'improvement': improvement
        }

    # Save all models
    model_path = Path(__file__).parent / 'multi_metric_models.pkl'
    joblib.dump({
        'models': models,
        'metrics': list(models.keys()),
        'training_count': len(training_data),
        'feature_names': feature_names
    }, model_path)

    print(f"\n💾 Multi-metric models saved: {model_path}")

    # Summary
    print(f"\n📈 TRAINING SUMMARY:")
    for metric, result in results.items():
        print(f"   {metric}: {result['improvement']:+.1f}% improvement")

    avg_improvement = np.mean([r['improvement'] for r in results.values()])
    print(f"   Average improvement: {avg_improvement:+.1f}%")

    return models, results

def generate_multi_metric_dataset(max_files=150):
    """Generate comprehensive dataset for multi-metric ML training"""

    print("🔬 MULTI-METRIC ML TRAINING PIPELINE")
    print("=" * 60)

    # Find training files
    training_dir = Path.home() / 'FlySight' / 'Training'
    csv_files = list(training_dir.glob('**/*.csv'))[:max_files]

    print(f"📁 Processing {len(csv_files)} files for comprehensive metrics...")

    # Process files
    training_data = []
    for i, filepath in enumerate(csv_files, 1):
        result = process_training_file_comprehensive(str(filepath))
        if result:
            training_data.append(result)

        if i % 25 == 0:
            print(f"   Processed: {i}/{len(csv_files)}, Success: {len(training_data)}")

    print(f"\n📊 Dataset Complete:")
    print(f"   Total processed: {len(csv_files)}")
    print(f"   Successful: {len(training_data)}")
    print(f"   Success rate: {len(training_data)/len(csv_files)*100:.1f}%")

    if training_data:
        # Show sample metrics distribution
        sample_metrics = training_data[0]['gswoop_metrics']
        print(f"\n📋 Sample metrics available:")
        for field_name in GswoopMetrics.__annotations__:
            value = getattr(sample_metrics, field_name)
            print(f"   {field_name}: {value}")

    return training_data

def main():
    """Main execution for multi-metric ML pipeline"""

    # Generate comprehensive dataset
    training_data = generate_multi_metric_dataset(max_files=100)  # Start with smaller set

    if len(training_data) > 10:
        # Train models for multiple metrics
        models, results = train_multi_metric_models(training_data)

        if models:
            print(f"\n🎉 MULTI-METRIC ML PIPELINE COMPLETE!")
            print(f"   Trained {len(models)} metric models")

            significant_improvements = [k for k, v in results.items() if v['improvement'] > 10]
            if significant_improvements:
                print(f"   ✅ Significant improvements in: {', '.join(significant_improvements)}")
            else:
                print(f"   ⚡ Moderate improvements across all metrics")
    else:
        print("❌ Insufficient training data for multi-metric ML models")

if __name__ == "__main__":
    main()