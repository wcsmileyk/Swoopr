#!/usr/bin/env python3
"""
ML Model Validation Suite
Compare ML predictions vs traditional algorithm vs gswoop ground truth
"""

import os
import sys
import django
import subprocess
import re
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager


@dataclass
class ValidationResult:
    """Result of validating a single flight"""
    filename: str
    gswoop_rotation: Optional[float]
    ml_rotation: Optional[float]
    traditional_rotation: Optional[float]
    ml_confidence: float
    ml_error: Optional[float]
    traditional_error: Optional[float]
    is_valid: bool
    error_message: str = ""

    def get_better_method(self) -> str:
        """Return which method is better"""
        if self.ml_error is None or self.traditional_error is None:
            return "unknown"
        return "ml" if self.ml_error < self.traditional_error else "traditional"


class MLValidator:
    """Validate ML model performance"""

    def __init__(self):
        self.fm = FlightManager()
        self.results = []

    def extract_gswoop_rotation(self, output: str) -> Optional[float]:
        """Parse rotation from gswoop output"""
        match = re.search(r'degrees of rotation:\s+(\d+)\s+deg\s+\((\w+)-hand\)', output)
        if match:
            degrees = float(match.group(1))
            direction = match.group(2)
            return degrees if direction == 'right' else -degrees
        return None

    def analyze_file(self, filepath: str) -> ValidationResult:
        """Analyze a single file with all methods"""
        filename = Path(filepath).name

        try:
            # Get gswoop ground truth
            print(f"  📊 Running gswoop...", end=" ", flush=True)
            result = subprocess.run(
                ['gswoop', '-m', '-i', filepath],
                capture_output=True,
                text=True,
                timeout=30,
                cwd='/tmp'
            )

            gswoop_rotation = self.extract_gswoop_rotation(result.stdout)
            if gswoop_rotation is None:
                return ValidationResult(
                    filename=filename,
                    gswoop_rotation=None,
                    ml_rotation=None,
                    traditional_rotation=None,
                    ml_confidence=0,
                    ml_error=None,
                    traditional_error=None,
                    is_valid=False,
                    error_message="gswoop failed to parse"
                )
            print(f"gswoop={gswoop_rotation:.0f}°", flush=True)

            # Analyze with Swoopr
            print(f"    🔍 Analyzing with Swoopr...", end=" ", flush=True)
            df, metadata = self.fm.read_flysight_file(filepath)
            landing_idx = self.fm.get_landing(df)

            try:
                flare_idx = self.fm.find_flare(df, landing_idx)
            except:
                flare_idx = self.fm.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = self.fm.find_max_speeds(df, flare_idx, landing_idx)

            # Traditional method
            dual_metrics = self.fm.calculate_dual_rotation_metrics(
                df, flare_idx, max_gspeed_idx, landing_idx
            )
            traditional_rotation = dual_metrics['full_swoop']['rotation']

            # ML method
            ml_rotation, ml_confidence, _ = self.fm.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

            print(f"ML={ml_rotation:.0f}° (conf={ml_confidence:.2f}), Traditional={traditional_rotation:.0f}°")

            # Calculate errors
            ml_error = abs(ml_rotation - gswoop_rotation) if ml_rotation else None
            traditional_error = abs(traditional_rotation - gswoop_rotation)

            return ValidationResult(
                filename=filename,
                gswoop_rotation=gswoop_rotation,
                ml_rotation=ml_rotation,
                traditional_rotation=traditional_rotation,
                ml_confidence=ml_confidence,
                ml_error=ml_error,
                traditional_error=traditional_error,
                is_valid=True
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                filename=filename,
                gswoop_rotation=None,
                ml_rotation=None,
                traditional_rotation=None,
                ml_confidence=0,
                ml_error=None,
                traditional_error=None,
                is_valid=False,
                error_message="gswoop timeout"
            )
        except Exception as e:
            return ValidationResult(
                filename=filename,
                gswoop_rotation=None,
                ml_rotation=None,
                traditional_rotation=None,
                ml_confidence=0,
                ml_error=None,
                traditional_error=None,
                is_valid=False,
                error_message=str(e)
            )

    def validate_directory(self, directory: str, max_files: Optional[int] = None):
        """Validate all CSV files in a directory"""
        path = Path(directory)
        csv_files = sorted(path.glob('**/*.csv'))

        if max_files:
            csv_files = csv_files[:max_files]

        print("\n" + "=" * 80)
        print("🤖 ML MODEL VALIDATION SUITE")
        print("=" * 80)
        print(f"\n📁 Found {len(csv_files)} CSV files to validate")
        print("Comparing: gswoop (ground truth) vs ML vs Traditional\n")

        self.results = []
        for i, filepath in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] {Path(filepath).name}")
            result = self.analyze_file(str(filepath))
            self.results.append(result)

            # Brief delay between gswoop calls
            time.sleep(0.1)

        self.generate_report()

    def generate_report(self):
        """Generate validation report"""
        print("\n" + "=" * 80)
        print("📋 VALIDATION REPORT")
        print("=" * 80)

        # Statistics
        valid_results = [r for r in self.results if r.is_valid]
        invalid_results = [r for r in self.results if not r.is_valid]

        print(f"\n✅ Valid analyses: {len(valid_results)}/{len(self.results)}")
        print(f"❌ Invalid analyses: {len(invalid_results)}/{len(self.results)}")

        if invalid_results:
            print(f"\nInvalid files:")
            for r in invalid_results:
                print(f"  - {r.filename}: {r.error_message}")

        if not valid_results:
            print("\n⚠️  No valid results to analyze!")
            return

        # ML Analysis
        ml_errors = [r.ml_error for r in valid_results if r.ml_error is not None]
        traditional_errors = [r.traditional_error for r in valid_results]

        print(f"\n📊 ERROR ANALYSIS")
        print(f"  Traditional Algorithm:")
        print(f"    Mean error: {np.mean(traditional_errors):.1f}°")
        print(f"    Median error: {np.median(traditional_errors):.1f}°")
        print(f"    Std dev: {np.std(traditional_errors):.1f}°")
        print(f"    Max error: {np.max(traditional_errors):.1f}°")

        if ml_errors:
            print(f"\n  ML Model:")
            print(f"    Mean error: {np.mean(ml_errors):.1f}°")
            print(f"    Median error: {np.median(ml_errors):.1f}°")
            print(f"    Std dev: {np.std(ml_errors):.1f}°")
            print(f"    Max error: {np.max(ml_errors):.1f}°")

            # Comparison
            ml_better = sum(1 for r in valid_results if r.ml_error and r.ml_error < r.traditional_error)
            traditional_better = sum(1 for r in valid_results if r.ml_error and r.ml_error >= r.traditional_error)

            print(f"\n🏆 HEAD-TO-HEAD:")
            print(f"  ML better: {ml_better}/{len(valid_results)} ({ml_better/len(valid_results)*100:.1f}%)")
            print(f"  Traditional better: {traditional_better}/{len(valid_results)} ({traditional_better/len(valid_results)*100:.1f}%)")

            # Average improvement
            improvements = []
            for r in valid_results:
                if r.ml_error is not None:
                    improvement = ((r.traditional_error - r.ml_error) / r.traditional_error) * 100
                    improvements.append(improvement)

            if improvements:
                print(f"  Average improvement: {np.mean(improvements):+.1f}%")
                print(f"  Median improvement: {np.median(improvements):+.1f}%")

        # Confidence Analysis
        print(f"\n🎯 ML CONFIDENCE ANALYSIS:")
        confidences = [r.ml_confidence for r in valid_results if r.ml_rotation is not None]
        if confidences:
            print(f"  Mean confidence: {np.mean(confidences):.2f}")
            print(f"  Median confidence: {np.median(confidences):.2f}")
            print(f"  Min confidence: {np.min(confidences):.2f}")
            print(f"  Max confidence: {np.max(confidences):.2f}")

            # How often ML is used
            high_conf = sum(1 for c in confidences if c > 0.4)
            print(f"  High confidence (>0.4): {high_conf}/{len(confidences)} ({high_conf/len(confidences)*100:.1f}%)")

        # Detailed results table
        print(f"\n📈 DETAILED RESULTS:")
        print(f"{'File':<30} {'gswoop':<10} {'ML':<10} {'ML Conf':<8} {'Traditional':<12} {'Winner':<12}")
        print("-" * 90)

        for r in sorted(valid_results, key=lambda x: x.filename)[:20]:  # Show first 20
            if r.gswoop_rotation is not None:
                ml_val = f"{r.ml_rotation:.0f}°" if r.ml_rotation else "N/A"
                winner = r.get_better_method().upper()
                print(
                    f"{r.filename:<30} "
                    f"{r.gswoop_rotation:<10.0f} "
                    f"{ml_val:<10} "
                    f"{r.ml_confidence:<8.2f} "
                    f"{r.traditional_rotation:<12.0f} "
                    f"{winner:<12}"
                )

        # Save results to CSV
        csv_path = Path(__file__).parent / 'validation_results.csv'
        df = pd.DataFrame([
            {
                'filename': r.filename,
                'gswoop_rotation': r.gswoop_rotation,
                'ml_rotation': r.ml_rotation,
                'ml_confidence': r.ml_confidence,
                'traditional_rotation': r.traditional_rotation,
                'ml_error': r.ml_error,
                'traditional_error': r.traditional_error,
                'ml_better': r.ml_error < r.traditional_error if (r.ml_error and r.traditional_error) else None,
            }
            for r in valid_results
        ])
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Results saved to: {csv_path}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if ml_errors:
            avg_ml_error = np.mean(ml_errors)
            avg_trad_error = np.mean(traditional_errors)

            if avg_ml_error < avg_trad_error * 0.9:
                print(f"  ✅ ML is significantly better! Keep it and celebrate!")
            elif avg_ml_error < avg_trad_error:
                print(f"  ⚠️  ML is slightly better but within noise. Consider retraining with more data.")
            else:
                print(f"  ❌ ML is worse than traditional. Options:")
                print(f"     1. Retrain model with better data")
                print(f"     2. Try different features")
                print(f"     3. Disable ML (use traditional only)")
        else:
            print(f"  ⚠️  No ML predictions available. Check feature extraction.")


def main():
    """Main validation entry point"""
    validator = MLValidator()

    # Try common locations for training data
    locations = [
        Path.home() / 'FlySight' / 'Training',
        Path('/home/smiley/PycharmProjects/Swoopr/sample_tracks'),
        Path.cwd() / 'sample_tracks',
    ]

    for loc in locations:
        if loc.exists():
            print(f"🎯 Found training data at: {loc}")
            # Ask user how many to validate
            import sys
            if len(sys.argv) > 1:
                max_files = int(sys.argv[1])
            else:
                max_files = None  # All files

            validator.validate_directory(str(loc), max_files=max_files)
            return

    print("❌ Could not find training data directory!")
    print("Tried:")
    for loc in locations:
        print(f"  - {loc}")
    print("\nUsage: python ml_validation_suite.py [max_files]")
    print("Example: python ml_validation_suite.py 50")


if __name__ == "__main__":
    main()
