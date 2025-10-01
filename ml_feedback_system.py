#!/usr/bin/env python3
"""
ML Feedback System - Correct ML predictions and retrain models
"""

import os
import sys
import django
import subprocess
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

def correct_ml_prediction(flight_file, correct_rotation, retrain=True):
    """
    Correct an ML prediction by adding it to training data and optionally retraining

    Args:
        flight_file: Path to the FlySight CSV file
        correct_rotation: The correct rotation value (from gswoop or manual verification)
        retrain: Whether to retrain the model after adding the correction
    """

    print(f"🔧 CORRECTING ML PREDICTION")
    print("=" * 50)
    print(f"File: {flight_file}")
    print(f"Correct rotation: {correct_rotation}°")

    # 1. Add to training corrections file
    corrections_file = Path(__file__).parent / 'ml_training_corrections.csv'

    # Create headers if file doesn't exist
    if not corrections_file.exists():
        with open(corrections_file, 'w') as f:
            f.write("filename,correct_rotation,correction_date,notes\n")

    # Add this correction
    from datetime import datetime
    with open(corrections_file, 'a') as f:
        filename = Path(flight_file).name
        date = datetime.now().strftime('%Y-%m-%d')
        f.write(f"{filename},{correct_rotation},{date},Manual correction\n")

    print(f"✅ Added correction to {corrections_file}")

    if retrain:
        print(f"\n🧠 Retraining models with correction...")

        # 2. Retrain the rotation model
        try:
            # Use the simple ML pipeline to retrain with corrections
            from simple_ml_pipeline import generate_simple_dataset, train_ml_model

            # Load existing training data
            training_data = generate_simple_dataset(max_files=100)

            # Apply corrections to the training data
            corrections_df = pd.read_csv(corrections_file)

            for i, example in enumerate(training_data):
                filename = example['filename']
                correction_row = corrections_df[corrections_df['filename'] == filename]
                if not correction_row.empty:
                    # Update the gswoop rotation with our correction
                    old_rotation = example['gswoop_rotation']
                    new_rotation = correction_row.iloc[0]['correct_rotation']
                    training_data[i]['gswoop_rotation'] = new_rotation
                    print(f"   Corrected {filename}: {old_rotation}° → {new_rotation}°")

            # Retrain the model
            if len(training_data) > 20:
                model, features, improvement = train_ml_model(training_data)
                print(f"✅ Model retrained with {improvement:+.1f}% improvement")
            else:
                print("❌ Not enough training data for retraining")

        except Exception as e:
            print(f"❌ Retraining failed: {e}")

    print(f"\n📊 Recommendation: Upload the corrected flight again to see improved predictions")

def bulk_correct_from_gswoop(training_dir, max_files=50):
    """
    Bulk correction: run gswoop on training files and correct any major discrepancies
    """

    print(f"🔄 BULK CORRECTION FROM GSWOOP")
    print("=" * 50)

    from flights.flight_manager import FlightManager

    training_path = Path(training_dir)
    csv_files = list(training_path.glob('**/*.csv'))[:max_files]

    corrections = []
    manager = FlightManager()

    for i, filepath in enumerate(csv_files, 1):
        try:
            # Get our ML prediction
            df, metadata = manager.read_flysight_file(str(filepath))
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)
            ml_rotation, ml_confidence, ml_method = manager.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

            # Get gswoop ground truth
            result = subprocess.run(['gswoop', '-i', str(filepath)],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                rotation_match = re.search(r'degrees of rotation:\s+(\d+) deg \((\w+)-hand\)', result.stdout)
                if rotation_match:
                    degrees = float(rotation_match.group(1))
                    direction = rotation_match.group(2)
                    gswoop_rotation = degrees if direction == 'right' else -degrees

                    # Check if ML prediction is significantly off
                    if ml_rotation is not None:
                        diff = abs(ml_rotation - gswoop_rotation)

                        if diff > 100:  # Significant error
                            corrections.append({
                                'filename': filepath.name,
                                'ml_prediction': ml_rotation,
                                'gswoop_truth': gswoop_rotation,
                                'difference': diff,
                                'confidence': ml_confidence
                            })

                            print(f"❌ {filepath.name}: ML={ml_rotation:.0f}° vs Gswoop={gswoop_rotation:.0f}° (Δ{diff:.0f}°)")

            if i % 10 == 0:
                print(f"   Processed: {i}/{len(csv_files)}")

        except Exception as e:
            continue

    print(f"\n📋 Found {len(corrections)} flights needing correction")

    if corrections:
        # Save corrections
        corrections_df = pd.DataFrame(corrections)
        corrections_file = Path(__file__).parent / 'bulk_corrections.csv'
        corrections_df.to_csv(corrections_file, index=False)

        print(f"💾 Saved corrections to {corrections_file}")
        print(f"📊 Top corrections needed:")

        # Show worst cases
        worst_cases = corrections_df.nlargest(5, 'difference')
        for _, row in worst_cases.iterrows():
            print(f"   {row['filename']}: ML={row['ml_prediction']:.0f}° vs Truth={row['gswoop_truth']:.0f}° (Δ{row['difference']:.0f}°)")

def interactive_correction():
    """Interactive correction system"""

    print(f"🎯 INTERACTIVE ML CORRECTION")
    print("=" * 50)

    while True:
        flight_file = input("Enter flight file path (or 'quit'): ").strip()

        if flight_file.lower() == 'quit':
            break

        if not Path(flight_file).exists():
            print("❌ File not found")
            continue

        # Analyze the flight
        try:
            from flights.flight_manager import FlightManager

            manager = FlightManager()
            df, metadata = manager.read_flysight_file(flight_file)
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Get current predictions
            traditional_rotation, _, _, _ = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)
            ml_rotation, ml_confidence, _ = manager.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

            # Get gswoop
            result = subprocess.run(['gswoop', '-i', flight_file],
                                  capture_output=True, text=True, timeout=30)

            gswoop_rotation = None
            if result.returncode == 0:
                rotation_match = re.search(r'degrees of rotation:\s+(\d+) deg \((\w+)-hand\)', result.stdout)
                if rotation_match:
                    degrees = float(rotation_match.group(1))
                    direction = rotation_match.group(2)
                    gswoop_rotation = degrees if direction == 'right' else -degrees

            print(f"\n📊 Current Predictions:")
            print(f"   Traditional: {traditional_rotation:.1f}°")
            print(f"   ML: {ml_rotation:.1f}° (confidence: {ml_confidence:.2f})")
            if gswoop_rotation:
                print(f"   Gswoop: {gswoop_rotation:.1f}°")

            # Ask for correction
            correct_value = input("Enter correct rotation (or 'skip'): ").strip()

            if correct_value.lower() != 'skip':
                try:
                    correct_rotation = float(correct_value)
                    correct_ml_prediction(flight_file, correct_rotation, retrain=True)
                except ValueError:
                    print("❌ Invalid rotation value")

        except Exception as e:
            print(f"❌ Error analyzing flight: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "bulk":
            # Bulk correction mode
            training_dir = sys.argv[2] if len(sys.argv) > 2 else str(Path.home() / 'FlySight' / 'Training')
            bulk_correct_from_gswoop(training_dir)
        elif sys.argv[1] == "correct":
            # Single file correction
            if len(sys.argv) >= 4:
                correct_ml_prediction(sys.argv[2], float(sys.argv[3]))
            else:
                print("Usage: python ml_feedback_system.py correct <file> <correct_rotation>")
        else:
            print("Usage: python ml_feedback_system.py [bulk|correct|interactive]")
    else:
        # Interactive mode
        interactive_correction()