#!/usr/bin/env python3
"""
Machine Learning Training Pipeline for Swoop Rotation Analysis
Uses dual metrics system with gswoop ground truth for training
"""

import os
import sys
import django
import subprocess
import re
import numpy as np
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

@dataclass
class TrainingExample:
    """Single training example with features and ground truth"""
    filename: str

    # Ground truth (gswoop)
    gswoop_rotation: float
    gswoop_turn_time: float
    gswoop_turn_start_alt: float
    gswoop_max_vspeed_alt: float

    # Our algorithm results
    full_swoop_rotation: float
    full_swoop_confidence: float
    full_swoop_method: str
    full_swoop_intended: int

    turn_segment_rotation: Optional[float]
    turn_segment_confidence: Optional[float]
    turn_segment_method: Optional[str]
    turn_segment_intended: Optional[int]

    # Features for ML
    features: Dict[str, float]

    # Quality metrics
    difference_from_gswoop: float
    is_reliable: bool

class MLTrainingPipeline:
    """Complete ML training pipeline for swoop rotation analysis"""

    def __init__(self):
        self.manager = FlightManager()
        self.training_examples = []
        self.lock = threading.Lock()

    def parse_gswoop_output(self, output: str) -> Optional[Dict]:
        """Parse gswoop output to extract ground truth data"""
        data = {}

        patterns = {
            'turn_start_alt': r'initiated turn:\s+(\d+)\s+ft AGL',
            'max_vspeed_alt': r'max vertical speed:\s+(\d+)\s+ft AGL',
            'max_vspeed_mph': r'max vertical speed:.*\((\d+\.\d+)\s+mph\)',
            'turn_time': r'time to execute turn:\s+(\d+\.\d+)\s+sec',
            'rotation_deg': r'degrees of rotation:\s+(\d+)\s+deg',
            'rotation_dir': r'degrees of rotation:.*\((\w+)-hand\)',
            'entry_speed': r'entry gate speed:\s+(\d+\.\d+)\s+mph',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                try:
                    data[key] = float(match.group(1))
                except:
                    data[key] = match.group(1)

        # Calculate signed rotation
        if 'rotation_deg' in data and 'rotation_dir' in data:
            degrees = data['rotation_deg']
            direction = data['rotation_dir']
            data['rotation'] = degrees if direction == 'right' else -degrees
            return data

        return None

    def extract_features(self, df, flare_idx, max_gspeed_idx, landing_idx, dual_metrics) -> Dict[str, float]:
        """Extract comprehensive features for ML training"""
        features = {}

        # Basic flight characteristics
        features['flight_duration'] = len(df) * 0.2
        features['total_altitude_loss'] = (df.iloc[0]['AGL'] - df.iloc[-1]['AGL']) / 0.3048

        # Turn segment analysis
        if flare_idx is not None and max_gspeed_idx is not None:
            turn_data = df[flare_idx:max_gspeed_idx+1]

            features['turn_duration'] = len(turn_data) * 0.2
            features['turn_points'] = len(turn_data)

            # Altitude features
            features['entry_altitude'] = df.iloc[flare_idx]['AGL'] / 0.3048
            features['max_gspeed_altitude'] = df.iloc[max_gspeed_idx]['AGL'] / 0.3048
            features['altitude_loss_in_turn'] = (df.iloc[flare_idx]['AGL'] - df.iloc[max_gspeed_idx]['AGL']) / 0.3048

            # Speed features
            features['entry_speed'] = df.iloc[flare_idx]['gspeed'] * 2.23694  # mph
            features['max_vspeed'] = abs(df.iloc[max_gspeed_idx]['velD']) * 2.23694  # mph
            features['max_gspeed'] = df.iloc[max_gspeed_idx]['gspeed'] * 2.23694  # mph

            # Heading analysis
            headings = turn_data['heading'].values
            if len(headings) >= 2:
                features['heading_start'] = headings[0]
                features['heading_end'] = headings[-1]

                # Net heading change
                net_change = headings[-1] - headings[0]
                while net_change > 180:
                    net_change -= 360
                while net_change < -180:
                    net_change += 360
                features['net_heading_change'] = net_change

                # Heading statistics
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
                    # Default values for missing heading data
                    for key in ['avg_turn_rate', 'max_turn_rate', 'turn_rate_std', 'total_heading_change', 'direction_consistency']:
                        features[key] = 0

            # GPS accuracy during turn
            if 'hAcc' in turn_data.columns:
                features['avg_horizontal_accuracy'] = turn_data['hAcc'].mean()
                features['max_horizontal_accuracy'] = turn_data['hAcc'].max()
            else:
                features['avg_horizontal_accuracy'] = 5.0  # Default assumption
                features['max_horizontal_accuracy'] = 10.0

        # Dual metrics features
        if 'full_swoop' in dual_metrics:
            fs = dual_metrics['full_swoop']
            features['fs_rotation'] = fs['rotation']
            features['fs_confidence'] = fs['confidence']
            features['fs_intended'] = fs['intended_turn']
            features['fs_duration'] = fs['duration']

        if 'turn_segment' in dual_metrics:
            ts = dual_metrics['turn_segment']
            features['ts_rotation'] = ts['rotation']
            features['ts_confidence'] = ts['confidence']
            features['ts_intended'] = ts['intended_turn']
            features['ts_duration'] = ts['duration']
            features['ts_start_alt'] = ts['start_alt']
            features['ts_end_alt'] = ts['end_alt']
        else:
            # Default values when turn segment not available
            for key in ['ts_rotation', 'ts_confidence', 'ts_intended', 'ts_duration', 'ts_start_alt', 'ts_end_alt']:
                features[key] = 0 if 'confidence' not in key else 0.1

        return features

    def process_single_file(self, filepath: str) -> Optional[TrainingExample]:
        """Process a single training file"""
        try:
            filename = Path(filepath).name

            # Run gswoop analysis
            result = subprocess.run(['gswoop', '-i', filepath],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                return None

            gswoop_data = self.parse_gswoop_output(result.stdout)
            if not gswoop_data or 'rotation' not in gswoop_data:
                return None

            # Analyze with our system
            df, metadata = self.manager.read_flysight_file(filepath)

            # Find key points
            landing_idx = self.manager.get_landing(df)

            try:
                flare_idx = self.manager.find_flare(df, landing_idx)
            except:
                flare_idx = self.manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = self.manager.find_max_speeds(df, flare_idx, landing_idx)

            # Calculate dual metrics
            dual_metrics = self.manager.calculate_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)

            if 'full_swoop' not in dual_metrics:
                return None

            # Extract features
            features = self.extract_features(df, flare_idx, max_gspeed_idx, landing_idx, dual_metrics)

            # Create training example
            fs = dual_metrics['full_swoop']
            ts = dual_metrics.get('turn_segment')

            # Calculate difference from gswoop
            difference = abs(fs['rotation'] - gswoop_data['rotation'])
            is_reliable = difference < 50  # Consider reliable if within 50°

            example = TrainingExample(
                filename=filename,

                # gswoop ground truth
                gswoop_rotation=gswoop_data['rotation'],
                gswoop_turn_time=gswoop_data.get('turn_time', 0),
                gswoop_turn_start_alt=gswoop_data.get('turn_start_alt', 0),
                gswoop_max_vspeed_alt=gswoop_data.get('max_vspeed_alt', 0),

                # Our results
                full_swoop_rotation=fs['rotation'],
                full_swoop_confidence=fs['confidence'],
                full_swoop_method=fs['method'],
                full_swoop_intended=fs['intended_turn'],

                turn_segment_rotation=ts['rotation'] if ts else None,
                turn_segment_confidence=ts['confidence'] if ts else None,
                turn_segment_method=ts['method'] if ts else None,
                turn_segment_intended=ts['intended_turn'] if ts else None,

                features=features,
                difference_from_gswoop=difference,
                is_reliable=is_reliable
            )

            return example

        except Exception as e:
            print(f"Error processing {Path(filepath).name}: {e}")
            return None

    def generate_training_dataset(self, max_files: Optional[int] = None) -> List[TrainingExample]:
        """Generate training dataset from all available files"""

        print("🤖 GENERATING ML TRAINING DATASET")
        print("=" * 60)

        # Find all training files
        training_dir = Path.home() / 'FlySight' / 'Training'
        csv_files = list(training_dir.glob('**/*.csv'))

        if max_files:
            csv_files = csv_files[:max_files]

        print(f"📁 Found {len(csv_files)} training files")
        print(f"🔄 Processing with {min(8, len(csv_files))} parallel workers...")

        # Process files in parallel
        training_examples = []
        processed = 0
        successful = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(self.process_single_file, str(f)): f for f in csv_files}

            # Collect results
            for future in as_completed(future_to_file):
                processed += 1
                filepath = future_to_file[future]

                try:
                    example = future.result()
                    if example:
                        training_examples.append(example)
                        successful += 1

                    if processed % 50 == 0:
                        print(f"   Processed: {processed}/{len(csv_files)}, Successful: {successful}")

                except Exception as e:
                    print(f"   Error with {filepath.name}: {e}")

        print(f"\n📊 Dataset Generation Complete:")
        print(f"   Total files processed: {processed}")
        print(f"   Successful extractions: {successful}")
        print(f"   Success rate: {successful/processed*100:.1f}%")

        # Analyze dataset quality
        if training_examples:
            reliable_count = sum(1 for ex in training_examples if ex.is_reliable)
            avg_difference = np.mean([ex.difference_from_gswoop for ex in training_examples])

            print(f"   Reliable examples (<50° diff): {reliable_count}/{successful} ({reliable_count/successful*100:.1f}%)")
            print(f"   Average difference from gswoop: {avg_difference:.1f}°")

        self.training_examples = training_examples
        return training_examples

    def save_training_dataset(self, output_file: str = 'ml_training_dataset.json'):
        """Save training dataset to file"""
        output_path = Path(__file__).parent / output_file

        # Convert to serializable format
        dataset = []
        for example in self.training_examples:
            dataset.append({
                'filename': example.filename,
                'gswoop_rotation': example.gswoop_rotation,
                'gswoop_turn_time': example.gswoop_turn_time,
                'gswoop_turn_start_alt': example.gswoop_turn_start_alt,
                'gswoop_max_vspeed_alt': example.gswoop_max_vspeed_alt,
                'full_swoop_rotation': example.full_swoop_rotation,
                'full_swoop_confidence': example.full_swoop_confidence,
                'full_swoop_method': example.full_swoop_method,
                'full_swoop_intended': example.full_swoop_intended,
                'turn_segment_rotation': example.turn_segment_rotation,
                'turn_segment_confidence': example.turn_segment_confidence,
                'turn_segment_method': example.turn_segment_method,
                'turn_segment_intended': example.turn_segment_intended,
                'features': example.features,
                'difference_from_gswoop': example.difference_from_gswoop,
                'is_reliable': bool(example.is_reliable)
            })

        with open(output_path, 'w') as f:
            json.dump({
                'dataset': dataset,
                'metadata': {
                    'total_examples': len(dataset),
                    'reliable_examples': sum(1 for ex in self.training_examples if ex.is_reliable),
                    'feature_count': len(self.training_examples[0].features) if self.training_examples else 0,
                    'generated_at': pd.Timestamp.now().isoformat()
                }
            }, f, indent=2)

        print(f"💾 Dataset saved to: {output_path}")
        return output_path

def main():
    """Main training pipeline execution"""

    pipeline = MLTrainingPipeline()

    # Generate training dataset (start with smaller subset for testing)
    print("Starting with 100 files for initial testing...")
    training_examples = pipeline.generate_training_dataset(max_files=100)

    if training_examples:
        # Save dataset
        output_file = pipeline.save_training_dataset()

        print(f"\n🎯 TRAINING DATASET READY!")
        print(f"   Examples: {len(training_examples)}")
        print(f"   Features per example: {len(training_examples[0].features)}")
        print(f"   Saved to: {output_file}")
        print(f"\n📋 Next Steps:")
        print(f"   ✅ Dataset generated with dual metrics")
        print(f"   ✅ gswoop ground truth integrated")
        print(f"   🔄 Ready for ML model training")
    else:
        print("❌ No training examples generated")

if __name__ == "__main__":
    main()