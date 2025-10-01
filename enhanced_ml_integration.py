#!/usr/bin/env python3
"""
Enhanced ML Integration - Multi-metric ML predictions for comprehensive swoop analysis
Integrates all the trained ML models into the production FlightManager
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

class MultiMetricMLFlightManager(FlightManager):
    """FlightManager with comprehensive ML-enhanced metric predictions"""

    def __init__(self):
        super().__init__()
        self.multi_metric_models = None
        self.multi_metric_loaded = False
        self.load_multi_metric_models()

    def load_multi_metric_models(self):
        """Load all trained multi-metric ML models"""
        try:
            model_path = Path(__file__).parent / 'multi_metric_models.pkl'
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.multi_metric_models = model_data['models']
                self.multi_metric_loaded = True
                print(f"✅ Multi-metric ML models loaded: {len(self.multi_metric_models)} metrics")
                print(f"   Available metrics: {', '.join(self.multi_metric_models.keys())}")
            else:
                print(f"⚠️  Multi-metric models not found: {model_path}")
        except Exception as e:
            print(f"❌ Error loading multi-metric models: {e}")

    def extract_ml_features_comprehensive(self, df, flare_idx, max_gspeed_idx):
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
        if len(turn_data) > 0:
            turn_total_speeds = np.sqrt(turn_data['velN']**2 + turn_data['velE']**2 + turn_data['velD']**2) * 2.23694
            features['max_total_speed'] = np.max(turn_total_speeds)
        else:
            features['max_total_speed'] = 0

        # Velocity component analysis
        features['max_vel_north'] = np.max(np.abs(turn_data['velN'])) * 2.23694 if len(turn_data) > 0 else 0
        features['max_vel_east'] = np.max(np.abs(turn_data['velE'])) * 2.23694 if len(turn_data) > 0 else 0
        features['max_vel_down'] = np.max(np.abs(turn_data['velD'])) * 2.23694 if len(turn_data) > 0 else 0

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
                for key in ['avg_turn_rate', 'max_turn_rate', 'turn_rate_std', 'total_heading_change', 'direction_consistency']:
                    features[key] = 0
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

    def predict_multi_metrics(self, df, flare_idx, max_gspeed_idx):
        """Predict multiple metrics using trained ML models"""
        if not self.multi_metric_loaded:
            return {}

        try:
            # Extract comprehensive features
            features = self.extract_ml_features_comprehensive(df, flare_idx, max_gspeed_idx)

            predictions = {}

            for metric_name, model_data in self.multi_metric_models.items():
                try:
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_names = model_data['feature_names']

                    # Create feature vector in correct order
                    feature_vector = np.array([[features.get(name, 0) for name in feature_names]])

                    # Scale features
                    feature_vector_scaled = scaler.transform(feature_vector)

                    # Predict
                    prediction = model.predict(feature_vector_scaled)[0]

                    # Store prediction with metadata
                    predictions[metric_name] = {
                        'value': prediction,
                        'confidence': min(1.0, max(0.3, 1.0 - model_data['mae'] / abs(prediction) if prediction != 0 else 0.5)),
                        'mae': model_data['mae'],
                        'improvement': model_data['improvement']
                    }

                except Exception as e:
                    print(f"Error predicting {metric_name}: {e}")
                    continue

            return predictions

        except Exception as e:
            print(f"Multi-metric prediction error: {e}")
            return {}

    def analyze_swoop_comprehensive(self, flight, df):
        """Comprehensive swoop analysis with multi-metric ML predictions"""

        # Run traditional analysis first
        traditional_result = self.analyze_swoop(flight, df)

        # If multi-metric models are available, add ML predictions
        if self.multi_metric_loaded and flight.analysis_successful:
            try:
                # Get key indices from the flight analysis
                landing_idx = flight.landing_idx
                flare_idx = flight.flare_idx
                max_gspeed_idx = flight.max_gspeed_idx

                if all(idx is not None for idx in [landing_idx, flare_idx, max_gspeed_idx]):
                    # Get multi-metric predictions
                    ml_predictions = self.predict_multi_metrics(df, flare_idx, max_gspeed_idx)

                    # Store ML predictions in flight model
                    self._store_ml_predictions(flight, ml_predictions)

                    # Update traditional_result with ML predictions
                    traditional_result.update({
                        'ml_predictions': ml_predictions,
                        'ml_metrics_count': len(ml_predictions)
                    })

            except Exception as e:
                print(f"Error in comprehensive ML analysis: {e}")

        return traditional_result

    def _store_ml_predictions(self, flight, predictions):
        """Store ML predictions in flight model"""
        from django.utils import timezone

        # Store timing predictions
        if 'time_to_execute_turn' in predictions:
            flight.ml_turn_time = predictions['time_to_execute_turn']['value']
            flight.ml_turn_time_confidence = predictions['time_to_execute_turn']['confidence']

        if 'time_during_rollout' in predictions:
            flight.ml_rollout_time = predictions['time_during_rollout']['value']
            flight.ml_rollout_time_confidence = predictions['time_during_rollout']['confidence']

        if 'time_aloft_during_swoop' in predictions:
            flight.ml_swoop_time = predictions['time_aloft_during_swoop']['value']
            flight.ml_swoop_time_confidence = predictions['time_aloft_during_swoop']['confidence']

        # Store distance predictions
        if 'distance_to_stop' in predictions:
            flight.ml_distance_to_stop = predictions['distance_to_stop']['value']
            flight.ml_distance_confidence = predictions['distance_to_stop']['confidence']

        if 'touchdown_estimate' in predictions:
            flight.ml_touchdown_distance = predictions['touchdown_estimate']['value']
            flight.ml_touchdown_confidence = predictions['touchdown_estimate']['confidence']

        if 'touchdown_speed_mph' in predictions:
            flight.ml_touchdown_speed = predictions['touchdown_speed_mph']['value']
            flight.ml_touchdown_speed_confidence = predictions['touchdown_speed_mph']['confidence']

        # Store entry gate prediction
        if 'entry_gate_speed_mph' in predictions:
            flight.ml_entry_speed = predictions['entry_gate_speed_mph']['value']
            flight.ml_entry_speed_confidence = predictions['entry_gate_speed_mph']['confidence']

        # Position metrics
        if 'turn_init_back' in predictions:
            flight.ml_turn_init_back = predictions['turn_init_back']['value']
        if 'turn_init_offset' in predictions:
            flight.ml_turn_init_offset = predictions['turn_init_offset']['value']

        # Store metadata
        flight.ml_predictions_count = len(predictions)
        flight.ml_predictions_updated_at = timezone.now()

        # Save the flight with new ML data
        flight.save()

def test_comprehensive_ml():
    """Test the comprehensive multi-metric ML system"""

    print("🔬 TESTING COMPREHENSIVE ML SYSTEM")
    print("=" * 60)

    # Test with sample file
    test_file = "/home/smiley/PycharmProjects/Swoopr/sample_tracks/25-07-07-sw3.csv"

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return

    # Initialize comprehensive ML manager
    comprehensive_manager = MultiMetricMLFlightManager()

    try:
        # Load and analyze file
        df, metadata = comprehensive_manager.read_flysight_file(test_file)

        # Find key points
        landing_idx = comprehensive_manager.get_landing(df)

        try:
            flare_idx = comprehensive_manager.find_flare(df, landing_idx)
        except:
            flare_idx = comprehensive_manager.find_turn_start_fallback(df, landing_idx)

        max_vspeed_idx, max_gspeed_idx = comprehensive_manager.find_max_speeds(df, flare_idx, landing_idx)

        print(f"📁 Test file: {os.path.basename(test_file)}")
        print(f"   GPS points: {len(df)}")

        # Get multi-metric predictions
        ml_predictions = comprehensive_manager.predict_multi_metrics(df, flare_idx, max_gspeed_idx)

        print(f"\n🤖 ML MULTI-METRIC PREDICTIONS:")
        print(f"   Models loaded: {comprehensive_manager.multi_metric_loaded}")
        print(f"   Predictions available: {len(ml_predictions)}")

        for metric, data in ml_predictions.items():
            value = data['value']
            confidence = data['confidence']
            improvement = data['improvement']

            # Format based on metric type
            if 'time' in metric:
                unit = 'sec'
                formatted_value = f"{value:.2f}"
            elif 'speed' in metric:
                unit = 'mph'
                formatted_value = f"{value:.1f}"
            elif 'distance' in metric or 'back' in metric or 'offset' in metric:
                unit = 'ft'
                formatted_value = f"{value:.0f}"
            elif 'rotation' in metric:
                unit = '°'
                formatted_value = f"{value:.0f}"
            else:
                unit = ''
                formatted_value = f"{value:.2f}"

            print(f"   {metric}: {formatted_value}{unit} (conf: {confidence:.2f}, +{improvement:.0f}%)")

        print(f"\n✅ Comprehensive ML System Operational!")

    except Exception as e:
        print(f"❌ Error in comprehensive ML test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_ml()