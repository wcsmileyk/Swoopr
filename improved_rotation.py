#!/usr/bin/env python3
"""
Improved rotation detection algorithm based on analysis of sample data
"""

import numpy as np
import pandas as pd
# from scipy import signal  # Not needed

def improved_rotation_detection(df, flare_idx, max_gspeed_idx):
    """
    Enhanced rotation detection with multiple validation methods
    """

    # Extract turn data
    turn_data = df[flare_idx:max_gspeed_idx+1].copy()
    headings = turn_data['heading'].values
    times = turn_data['t_s'].values

    if len(headings) < 3:
        return 270.0, "default", 0.5  # Default fallback

    # Method 1: Smoothed heading approach to reduce GPS noise
    def smooth_headings(headings, window_size=5):
        """Smooth headings using a moving average, handling 360° wraparound"""
        # Convert to complex representation to handle wraparound
        complex_headings = np.exp(1j * np.deg2rad(headings))

        # Smooth in complex domain
        if len(complex_headings) >= window_size:
            # Use a simple moving average
            smoothed_complex = np.convolve(complex_headings,
                                         np.ones(window_size)/window_size,
                                         mode='same')
        else:
            smoothed_complex = complex_headings

        # Convert back to angles
        smoothed_headings = np.rad2deg(np.angle(smoothed_complex))
        smoothed_headings = (smoothed_headings + 360) % 360

        return smoothed_headings

    smooth_hdg = smooth_headings(headings)

    # Method 2: Progressive heading tracking with outlier rejection
    def progressive_heading_analysis(headings):
        """Track heading changes progressively, rejecting outliers"""

        if len(headings) < 2:
            return 0, 0

        changes = []
        prev_heading = headings[0]
        total_rotation = 0

        for heading in headings[1:]:
            # Calculate angular difference
            diff = heading - prev_heading

            # Normalize to [-180, 180]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            # Outlier rejection: reject changes > 120° (likely GPS noise)
            if abs(diff) <= 120:
                changes.append(diff)
                total_rotation += diff
                prev_heading = heading
            # For large jumps, keep the same heading (ignore the noise)

        return total_rotation, len(changes)

    # Method 3: Direction-consistent analysis
    def direction_consistent_analysis(headings):
        """Focus on the dominant turn direction"""

        changes = []
        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            if abs(diff) <= 120:  # Outlier rejection
                changes.append(diff)

        if not changes:
            return 0, 0

        # Determine dominant direction
        positive_sum = sum(c for c in changes if c > 0)
        negative_sum = sum(c for c in changes if c < 0)

        if abs(negative_sum) > abs(positive_sum):
            # Left turn dominant
            dominant_changes = [c for c in changes if c < 0]
            direction = -1
        else:
            # Right turn dominant
            dominant_changes = [c for c in changes if c > 0]
            direction = 1

        total_dominant = sum(dominant_changes)
        return total_dominant, direction

    # Apply all methods
    raw_rotation, _ = progressive_heading_analysis(headings)
    smooth_rotation, _ = progressive_heading_analysis(smooth_hdg)
    dominant_rotation, direction = direction_consistent_analysis(headings)

    # Method 4: Full rotation detection
    def detect_full_rotations(headings):
        """Detect if we've made full 360° rotations"""

        if len(headings) < 10:  # Need sufficient data
            return 0

        # Look for heading wrapping patterns
        wraps = 0

        # Track crossings of 0°/360° line
        for i in range(1, len(headings)):
            prev = headings[i-1]
            curr = headings[i]

            # Detect 360° -> 0° crossing (positive rotation)
            if prev > 270 and curr < 90:
                wraps += 1
            # Detect 0° -> 360° crossing (negative rotation)
            elif prev < 90 and curr > 270:
                wraps -= 1

        return wraps

    full_rotations = detect_full_rotations(headings)

    # Method 5: Turn classification and validation
    def classify_and_validate(rotations_dict):
        """Classify turn type and validate with multiple methods"""

        methods_agreement = {}

        for method_name, rotation in rotations_dict.items():
            abs_rotation = abs(rotation)

            # Add full rotations if detected
            if full_rotations != 0:
                abs_rotation += abs(full_rotations) * 360

            # Classify into standard turns
            if abs_rotation < 150:
                category = 90
            elif abs_rotation < 350:
                category = 270
            elif abs_rotation < 550:
                category = 450
            elif abs_rotation < 750:
                category = 630
            elif abs_rotation < 950:
                category = 810
            else:
                category = 990

            methods_agreement[method_name] = {
                'raw_rotation': rotation,
                'corrected_rotation': abs_rotation,
                'category': category,
                'confidence': calculate_confidence(abs_rotation, category)
            }

        return methods_agreement

    def calculate_confidence(rotation, category):
        """Calculate confidence based on distance from standard turn"""
        distance = abs(rotation - category)
        max_distance = 90  # Maximum acceptable distance
        confidence = max(0, (max_distance - distance) / max_distance)
        return confidence

    # Combine all methods
    rotation_methods = {
        'raw': raw_rotation,
        'smoothed': smooth_rotation,
        'dominant': dominant_rotation
    }

    classifications = classify_and_validate(rotation_methods)

    # Decision logic: pick the most confident method
    best_method = None
    best_confidence = 0

    for method_name, data in classifications.items():
        if data['confidence'] > best_confidence:
            best_confidence = data['confidence']
            best_method = method_name

    if best_method:
        result = classifications[best_method]
        final_rotation = result['corrected_rotation']
        intended_turn = result['category']
        confidence = result['confidence']

        # Apply direction
        if direction == -1:
            final_rotation = -final_rotation

        return final_rotation, intended_turn, confidence, best_method
    else:
        # Fallback to most conservative estimate
        return smooth_rotation, 270, 0.3, 'fallback'

def analyze_improved_algorithm():
    """Test the improved algorithm on sample data"""

    import os
    import django
    from pathlib import Path

    # Setup Django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
    django.setup()

    from flights.flight_manager import FlightManager

    sample_dir = Path('/home/smiley/PycharmProjects/Swoopr/sample_tracks')
    results = []

    print("🔬 TESTING IMPROVED ROTATION ALGORITHM")
    print("=" * 60)

    for csv_file in sample_dir.glob('*.csv'):
        if csv_file.name == 'gps_00545.csv':
            continue

        try:
            # Load file
            manager = FlightManager()
            df, metadata = manager.read_flysight_file(str(csv_file))

            # Find turn points
            landing_idx = manager.get_landing(df)
            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Compare algorithms
            old_rotation = manager.get_rotation(df, flare_idx, max_gspeed_idx)
            new_rotation, intended, confidence, method = improved_rotation_detection(df, flare_idx, max_gspeed_idx)

            print(f"\n📁 {csv_file.name}")
            print(f"   Old: {old_rotation:.1f}°")
            print(f"   New: {new_rotation:.1f}° → {intended}° (conf: {confidence:.2f}, method: {method})")

            improvement = "✅" if confidence > 0.7 else "⚠️" if confidence > 0.4 else "❌"
            print(f"   {improvement} Confidence: {confidence:.2f}")

            results.append({
                'file': csv_file.name,
                'old': old_rotation,
                'new': new_rotation,
                'intended': intended,
                'confidence': confidence,
                'method': method
            })

        except Exception as e:
            print(f"❌ Error with {csv_file.name}: {e}")

    # Summary
    if results:
        print(f"\n📊 SUMMARY:")
        print(f"   Files analyzed: {len(results)}")

        high_conf = [r for r in results if r['confidence'] > 0.7]
        medium_conf = [r for r in results if 0.4 <= r['confidence'] <= 0.7]
        low_conf = [r for r in results if r['confidence'] < 0.4]

        print(f"   High confidence (>0.7): {len(high_conf)}")
        print(f"   Medium confidence (0.4-0.7): {len(medium_conf)}")
        print(f"   Low confidence (<0.4): {len(low_conf)}")

        # Check improvement over old algorithm
        improvements = 0
        for r in results:
            old_distance = min([abs(abs(r['old']) - std) for std in [90, 270, 450, 630, 810, 990]])
            new_distance = abs(abs(r['new']) - r['intended'])
            if new_distance < old_distance:
                improvements += 1

        print(f"   Improvements over old: {improvements}/{len(results)} ({improvements/len(results)*100:.1f}%)")

if __name__ == "__main__":
    analyze_improved_algorithm()