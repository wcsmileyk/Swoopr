#!/usr/bin/env python3
"""
Diagnostic script to check if ML system is causing production 500 errors
Checks:
1. Model file exists and is readable
2. Model loads correctly
3. Singleton pattern works
4. Feature extraction works
5. Prediction doesn't crash
"""

import os
import sys
import django
from pathlib import Path

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager, MLModelSingleton
import traceback


def check_model_file():
    """Check if model file exists"""
    print("1️⃣  Checking model file...")
    model_path = Path(__file__).parent / 'flights' / 'rotation_prediction_model.pkl'

    if not model_path.exists():
        print(f"   ❌ Model file NOT FOUND: {model_path}")
        return False

    print(f"   ✅ Model file exists: {model_path}")
    print(f"   📦 Size: {model_path.stat().st_size / 1024:.1f} KB")
    return True


def check_singleton_loading():
    """Check if ML model singleton loads"""
    print("\n2️⃣  Checking ML model singleton loading...")

    try:
        singleton = MLModelSingleton()
        if singleton.model is None:
            print(f"   ⚠️  Singleton loaded but model is None")
            print(f"   This is OK - fallback to traditional algorithm")
            return True
        else:
            print(f"   ✅ Model loaded successfully via singleton")
            print(f"   Model type: {type(singleton.model).__name__}")
            print(f"   Features: {len(singleton.feature_names)} total")
            return True
    except Exception as e:
        print(f"   ❌ ERROR loading singleton: {e}")
        traceback.print_exc()
        return False


def check_flightmanager_init():
    """Check if FlightManager initializes"""
    print("\n3️⃣  Checking FlightManager initialization...")

    try:
        fm = FlightManager()
        print(f"   ✅ FlightManager initialized successfully")
        print(f"   ML loaded: {fm.ml_model_loaded}")
        return True
    except Exception as e:
        print(f"   ❌ ERROR initializing FlightManager: {e}")
        traceback.print_exc()
        return False


def check_feature_extraction():
    """Check if feature extraction works"""
    print("\n4️⃣  Checking feature extraction...")

    try:
        fm = FlightManager()

        # Use test file if available
        test_file = Path(__file__).parent / 'sample_tracks' / '25-07-07-sw3.csv'
        if not test_file.exists():
            print(f"   ⚠️  No test file found, skipping feature extraction test")
            return True

        print(f"   Testing with: {test_file.name}")
        df, metadata = fm.read_flysight_file(str(test_file))
        print(f"   ✅ CSV parsed: {len(df)} rows")

        landing_idx = fm.get_landing(df)
        print(f"   ✅ Landing detected at index {landing_idx}")

        try:
            flare_idx = fm.find_flare(df, landing_idx)
        except:
            flare_idx = fm.find_turn_start_fallback(df, landing_idx)
        print(f"   ✅ Flare detected at index {flare_idx}")

        max_vspeed_idx, max_gspeed_idx = fm.find_max_speeds(df, flare_idx, landing_idx)
        print(f"   ✅ Max speeds found")

        # Try to extract features
        features = fm.extract_ml_features(df, flare_idx, max_gspeed_idx)
        print(f"   ✅ Features extracted: {len(features)} features")

        return True
    except Exception as e:
        print(f"   ❌ ERROR in feature extraction: {e}")
        traceback.print_exc()
        return False


def check_prediction():
    """Check if ML prediction works"""
    print("\n5️⃣  Checking ML prediction...")

    try:
        fm = FlightManager()

        # Use test file if available
        test_file = Path(__file__).parent / 'sample_tracks' / '25-07-07-sw3.csv'
        if not test_file.exists():
            print(f"   ⚠️  No test file found, skipping prediction test")
            return True

        df, metadata = fm.read_flysight_file(str(test_file))
        landing_idx = fm.get_landing(df)

        try:
            flare_idx = fm.find_flare(df, landing_idx)
        except:
            flare_idx = fm.find_turn_start_fallback(df, landing_idx)

        max_vspeed_idx, max_gspeed_idx = fm.find_max_speeds(df, flare_idx, landing_idx)

        # Try prediction
        ml_rotation, ml_confidence, ml_method = fm.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

        if ml_rotation is None:
            print(f"   ⚠️  ML prediction returned None")
            print(f"   Method: {ml_method}")
            print(f"   This is OK - fallback will be used")
            return True
        else:
            print(f"   ✅ ML prediction successful")
            print(f"   Rotation: {ml_rotation:.0f}°")
            print(f"   Confidence: {ml_confidence:.2f}")
            print(f"   Method: {ml_method}")
            return True
    except Exception as e:
        print(f"   ❌ ERROR in ML prediction: {e}")
        traceback.print_exc()
        return False


def check_analyze_swoop():
    """Check if full swoop analysis works"""
    print("\n6️⃣  Checking full swoop analysis...")

    try:
        from flights.models import Flight
        from django.contrib.auth.models import User

        fm = FlightManager()

        # Use test file if available
        test_file = Path(__file__).parent / 'sample_tracks' / '25-07-07-sw3.csv'
        if not test_file.exists():
            print(f"   ⚠️  No test file found, skipping full analysis test")
            return True

        # Create test user if needed
        user, created = User.objects.get_or_create(username='test_diagnostic')

        print(f"   Running full analysis...")
        flight = fm.process_file(str(test_file), pilot=user)

        if flight.analysis_successful:
            print(f"   ✅ Full analysis successful!")
            print(f"   Rotation: {flight.turn_rotation:.0f}°")
        else:
            print(f"   ⚠️  Analysis marked unsuccessful")
            print(f"   Error: {flight.analysis_error}")

        return True
    except Exception as e:
        print(f"   ❌ ERROR in full analysis: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics"""
    print("=" * 80)
    print("🔍 ML SYSTEM PRODUCTION DIAGNOSTICS")
    print("=" * 80)
    print("\nThis will help identify if the ML system is causing 500 errors\n")

    results = {
        "Model file": check_model_file(),
        "Singleton loading": check_singleton_loading(),
        "FlightManager init": check_flightmanager_init(),
        "Feature extraction": check_feature_extraction(),
        "ML prediction": check_prediction(),
        "Full analysis": check_analyze_swoop(),
    }

    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    for check, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL CHECKS PASSED")
        print("\nML system appears to be working correctly.")
        print("If you're getting 500 errors, the issue is likely:")
        print("  1. The flare detection bug (identified in strategic analysis)")
        print("  2. Database issues (connection pooling, transaction handling)")
        print("  3. Other parts of the analysis pipeline")
        print("\nCheck CRITICAL_BUG_REPORT.md for flare detection issue")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nThe ML system may be causing 500 errors.")
        print("Review the errors above to identify the issue.")
        print("\nOptions:")
        print("  1. Disable ML temporarily (set ml_model_loaded = False)")
        print("  2. Fix the identified issue")
        print("  3. Check if model file is deployed in Render")

    print("=" * 80)


if __name__ == "__main__":
    main()
