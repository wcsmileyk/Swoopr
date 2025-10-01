#!/usr/bin/env python3
"""
Integration of ML model into FlightManager for production use
"""

import os
import sys
import django
import numpy as np
import joblib
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

class MLEnhancedFlightManager(FlightManager):
    """FlightManager with ML-enhanced rotation prediction"""

    def __init__(self):
        super().__init__()
        self.ml_model = None
        self.feature_names = None
        self.model_loaded = False
        self.load_ml_model()

    def load_ml_model(self):
        """Load the trained ML model"""
        try:
            model_path = Path(__file__).parent / 'rotation_prediction_model.pkl'
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.ml_model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.model_loaded = True
                print(f"✅ ML model loaded: {model_data['improvement']:+.1f}% improvement")
            else:
                print(f"⚠️  ML model not found: {model_path}")
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")

    def extract_ml_features(self, df, flare_idx, max_gspeed_idx):
        """Extract features for ML prediction (matching training format)"""
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

        return features

    def predict_ml_rotation(self, df, flare_idx, max_gspeed_idx):
        """Predict rotation using ML model"""
        if not self.model_loaded:
            return None, 0.0

        try:
            # Extract features
            features = self.extract_ml_features(df, flare_idx, max_gspeed_idx)

            # Create feature vector in correct order
            feature_vector = np.array([[features[name] for name in self.feature_names]])

            # Predict
            ml_rotation = self.ml_model.predict(feature_vector)[0]

            # Calculate confidence based on how well features match training data
            # Simple confidence calculation - can be improved
            confidence = min(1.0, max(0.3, 1.0 - abs(ml_rotation) / 1000))

            return ml_rotation, confidence

        except Exception as e:
            print(f"ML prediction error: {e}")
            return None, 0.0

    def get_rotation_with_ml(self, df, flare_idx, max_gspeed_idx):
        """Calculate rotation with ML enhancement"""

        # Get our traditional algorithm result
        traditional_rotation, intended_turn, traditional_confidence, method = self.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

        # Get ML prediction
        ml_rotation, ml_confidence = self.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

        if ml_rotation is not None and ml_confidence > 0.5:
            # Use ML prediction if confident

            # Classify ML prediction to intended turn
            abs_rotation = abs(ml_rotation)
            if abs_rotation < 150:
                ml_intended = 90
            elif abs_rotation < 350:
                ml_intended = 270
            elif abs_rotation < 550:
                ml_intended = 450
            elif abs_rotation < 750:
                ml_intended = 630
            elif abs_rotation < 950:
                ml_intended = 810
            else:
                ml_intended = 990

            return ml_rotation, ml_intended, ml_confidence, "ml_enhanced"
        else:
            # Fallback to traditional algorithm
            return traditional_rotation, intended_turn, traditional_confidence, method

    def calculate_triple_rotation_metrics(self, df, flare_idx, max_gspeed_idx, landing_idx):
        """
        Calculate three rotation metrics:
        1. Full swoop (traditional)
        2. Turn segment (gswoop-style)
        3. ML prediction (gswoop-aligned)
        """
        results = {}

        # 1. Traditional full swoop
        try:
            full_rotation, intended_turn, confidence, method = self.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)
            results['full_swoop'] = {
                'rotation': full_rotation,
                'intended_turn': intended_turn,
                'confidence': confidence,
                'method': method,
                'start_alt': df.iloc[flare_idx]['AGL'] / 0.3048,
                'end_alt': df.iloc[max_gspeed_idx]['AGL'] / 0.3048,
                'duration': (max_gspeed_idx - flare_idx) * 0.2,
            }
        except Exception as e:
            print(f"Full swoop calculation failed: {e}")

        # 2. ML-enhanced prediction
        try:
            ml_rotation, ml_intended, ml_confidence, ml_method = self.get_rotation_with_ml(df, flare_idx, max_gspeed_idx)
            results['ml_prediction'] = {
                'rotation': ml_rotation,
                'intended_turn': ml_intended,
                'confidence': ml_confidence,
                'method': ml_method,
                'start_alt': df.iloc[flare_idx]['AGL'] / 0.3048,
                'end_alt': df.iloc[max_gspeed_idx]['AGL'] / 0.3048,
                'duration': (max_gspeed_idx - flare_idx) * 0.2,
            }
        except Exception as e:
            print(f"ML prediction failed: {e}")

        # 3. Turn segment (if available)
        try:
            turn_segment_result = self._calculate_turn_segment_rotation(df, landing_idx)
            if turn_segment_result:
                results['turn_segment'] = turn_segment_result
        except Exception as e:
            print(f"Turn segment calculation failed: {e}")

        return results

def test_ml_integration():
    """Test the ML-enhanced flight manager"""

    print("🤖 TESTING ML-ENHANCED FLIGHT MANAGER")
    print("=" * 60)

    # Test with sample file
    test_file = "/home/smiley/PycharmProjects/Swoopr/sample_tracks/25-07-07-sw3.csv"

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return

    # Initialize ML-enhanced manager
    ml_manager = MLEnhancedFlightManager()

    try:
        # Load and analyze file
        df, metadata = ml_manager.read_flysight_file(test_file)

        # Find key points
        landing_idx = ml_manager.get_landing(df)

        try:
            flare_idx = ml_manager.find_flare(df, landing_idx)
        except:
            flare_idx = ml_manager.find_turn_start_fallback(df, landing_idx)

        max_vspeed_idx, max_gspeed_idx = ml_manager.find_max_speeds(df, flare_idx, landing_idx)

        print(f"📁 Test file: {os.path.basename(test_file)}")
        print(f"   GPS points: {len(df)}")

        # Calculate triple metrics
        triple_metrics = ml_manager.calculate_triple_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)

        print(f"\n🎯 TRIPLE ROTATION METRICS:")

        if 'full_swoop' in triple_metrics:
            fs = triple_metrics['full_swoop']
            print(f"   🔄 Traditional Full Swoop:")
            print(f"      Rotation: {fs['rotation']:.1f}° → {fs['intended_turn']}°")
            print(f"      Confidence: {fs['confidence']:.2f} ({fs['method']})")

        if 'ml_prediction' in triple_metrics:
            ml = triple_metrics['ml_prediction']
            print(f"   🤖 ML-Enhanced Prediction:")
            print(f"      Rotation: {ml['rotation']:.1f}° → {ml['intended_turn']}°")
            print(f"      Confidence: {ml['confidence']:.2f} ({ml['method']})")

        if 'turn_segment' in triple_metrics:
            ts = triple_metrics['turn_segment']
            print(f"   📊 Turn Segment:")
            print(f"      Rotation: {ts['rotation']:.1f}° → {ts['intended_turn']}°")
            print(f"      Confidence: {ts['confidence']:.2f} ({ts['method']})")

        # Compare results
        if 'full_swoop' in triple_metrics and 'ml_prediction' in triple_metrics:
            fs_rot = triple_metrics['full_swoop']['rotation']
            ml_rot = triple_metrics['ml_prediction']['rotation']
            difference = abs(fs_rot - ml_rot)

            print(f"\n📊 COMPARISON:")
            print(f"   Traditional vs ML difference: {difference:.1f}°")

            if difference < 30:
                print(f"   ✅ GOOD AGREEMENT between methods")
            elif difference < 100:
                print(f"   ⚡ MODERATE DIFFERENCE - ML providing correction")
            else:
                print(f"   ⚠️  LARGE DIFFERENCE - ML suggesting significant correction")

        print(f"\n✅ ML Integration Test Complete!")

    except Exception as e:
        print(f"❌ Error in ML integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_integration()