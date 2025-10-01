#!/usr/bin/env python3
"""
Fixed rotation detection algorithm aligned with gswoop methodology
Addresses direction interpretation and full rotation counting issues
"""

import os
import sys
import django
import numpy as np
import subprocess
import re

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

def fixed_rotation_detection(df, flare_idx, max_gspeed_idx):
    """
    Fixed rotation detection algorithm aligned with gswoop methodology

    Key fixes:
    1. Use net heading change as primary method (like gswoop appears to)
    2. Only count full rotations when net change is small but total change is large
    3. Apply consistent direction interpretation
    4. Conservative approach to avoid over-counting
    """

    # Extract turn data
    turn_data = df[flare_idx:max_gspeed_idx+1].copy()
    headings = turn_data['heading'].values

    if len(headings) < 3:
        return 270.0, 270, 0.5, "default"

    # Method 1: Calculate net heading change (primary approach like gswoop)
    start_heading = headings[0]
    end_heading = headings[-1]

    # Calculate net change with proper wraparound
    net_change = end_heading - start_heading
    while net_change > 180:
        net_change -= 360
    while net_change < -180:
        net_change += 360

    # Method 2: Calculate total angular distance traveled
    total_distance = 0
    valid_changes = []

    for i in range(1, len(headings)):
        diff = headings[i] - headings[i-1]

        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        # Filter out obvious GPS noise (>90° jumps)
        if abs(diff) <= 90:
            total_distance += abs(diff)
            valid_changes.append(diff)

    # Method 3: Detect if this is likely a multi-rotation turn
    # Key insight: Only add full rotations if there's strong evidence

    # Calculate turn consistency
    if valid_changes:
        # Check direction consistency
        positive_changes = sum(1 for c in valid_changes if c > 5)  # Right turns
        negative_changes = sum(1 for c in valid_changes if c < -5)  # Left turns
        total_directional = positive_changes + negative_changes

        if total_directional > 0:
            direction_consistency = abs(positive_changes - negative_changes) / total_directional
        else:
            direction_consistency = 0
    else:
        direction_consistency = 0

    # Determine if this is a multi-rotation turn
    # Conservative approach: only count extra rotations with strong evidence
    estimated_rotations = 0

    # Only consider multi-rotation if:
    # 1. Total distance is significantly larger than net change
    # 2. Direction is consistent (not oscillating)
    # 3. Turn is long enough to plausibly contain multiple rotations

    turn_duration = len(headings) * 0.2  # seconds

    if (total_distance > abs(net_change) + 180 and  # Significant extra distance
        direction_consistency > 0.7 and            # Consistent direction
        turn_duration > 8.0 and                    # Long enough turn
        total_distance > 450):                     # Substantial total rotation

        # Estimate additional full rotations conservatively
        extra_distance = total_distance - abs(net_change)
        estimated_rotations = int(extra_distance / 360)

        # Cap at reasonable maximum (most swoops are 1-3 rotations)
        estimated_rotations = min(estimated_rotations, 2)

    # Calculate final rotation
    base_rotation = abs(net_change) + (estimated_rotations * 360)

    # Apply direction based on net change
    if net_change < 0:
        final_rotation = -base_rotation
    else:
        final_rotation = base_rotation

    # Classify intended turn (standard increments)
    abs_rotation = abs(final_rotation)
    if abs_rotation < 150:
        intended_turn = 90
    elif abs_rotation < 350:
        intended_turn = 270
    elif abs_rotation < 550:
        intended_turn = 450
    elif abs_rotation < 750:
        intended_turn = 630
    elif abs_rotation < 950:
        intended_turn = 810
    else:
        intended_turn = 990

    # Calculate confidence based on multiple factors
    confidence = 0.5  # Base confidence

    # Higher confidence for consistent direction
    confidence += direction_consistency * 0.3

    # Higher confidence for reasonable turn duration
    if 3 <= turn_duration <= 15:
        confidence += 0.2

    # Lower confidence for estimated multi-rotations
    if estimated_rotations > 0:
        confidence -= 0.2

    # Higher confidence if rotation matches standard increment well
    distance_to_standard = min([abs(abs_rotation - std) for std in [90, 270, 450, 630, 810, 990]])
    if distance_to_standard < 30:
        confidence += 0.2

    confidence = max(0.1, min(1.0, confidence))

    method = f"fixed_v1_rotations_{estimated_rotations}"

    return final_rotation, intended_turn, confidence, method

def test_fixed_algorithm():
    """Test the fixed algorithm on problematic cases"""

    print("🔧 TESTING FIXED ROTATION ALGORITHM")
    print("=" * 60)

    # Test on cases where expert confirmed gswoop was correct
    test_cases = [
        ("~/FlySight/Training/25-02-20/25-02-20-sw1.csv", -275.0),  # Expert: left 270
        ("~/FlySight/Training/25-03-27/25-03-27-sw4.csv", -269.0),  # Expert: left 270
        ("~/FlySight/Training/24-10-12/24-10-12-sw3.csv", 102.0),   # Known good case
        ("~/FlySight/Training/25-07-04/25-07-04-sw3.csv", -93.0),   # Known good case
    ]

    manager = FlightManager()
    results = []

    for filepath, expected_gswoop in test_cases:
        filepath = os.path.expanduser(filepath)
        filename = os.path.basename(filepath)

        print(f"\n📁 Testing: {filename}")

        try:
            # Get gswoop result
            result = subprocess.run(['gswoop', '-i', filepath],
                                  capture_output=True, text=True, timeout=30)
            gswoop_rotation = parse_gswoop_rotation(result.stdout)

            # Analyze with our algorithms
            df, metadata = manager.read_flysight_file(filepath)
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Old algorithm
            old_rotation, old_intended, old_confidence, old_method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

            # New fixed algorithm
            new_rotation, new_intended, new_confidence, new_method = fixed_rotation_detection(df, flare_idx, max_gspeed_idx)

            # Compare results
            old_diff = abs(old_rotation - gswoop_rotation) if gswoop_rotation else 999
            new_diff = abs(new_rotation - gswoop_rotation) if gswoop_rotation else 999

            print(f"   📊 gswoop:      {gswoop_rotation:.1f}°")
            print(f"   🔄 Old algo:    {old_rotation:.1f}° (diff: {old_diff:.1f}°)")
            print(f"   ✨ Fixed algo:  {new_rotation:.1f}° (diff: {new_diff:.1f}°, conf: {new_confidence:.2f})")

            improvement = old_diff - new_diff
            if improvement > 20:
                print(f"   ✅ SIGNIFICANT IMPROVEMENT: {improvement:.1f}° better")
            elif improvement > 0:
                print(f"   ⚡ IMPROVEMENT: {improvement:.1f}° better")
            elif improvement < -20:
                print(f"   ❌ REGRESSION: {abs(improvement):.1f}° worse")
            else:
                print(f"   ➡️  Similar: {improvement:.1f}° difference")

            results.append({
                'filename': filename,
                'gswoop': gswoop_rotation,
                'old_rotation': old_rotation,
                'new_rotation': new_rotation,
                'old_diff': old_diff,
                'new_diff': new_diff,
                'improvement': improvement
            })

        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Summary
    if results:
        print(f"\n📊 SUMMARY:")
        print(f"   Test cases: {len(results)}")

        total_old_error = sum(r['old_diff'] for r in results)
        total_new_error = sum(r['new_diff'] for r in results)
        avg_improvement = sum(r['improvement'] for r in results) / len(results)

        print(f"   Average old error: {total_old_error / len(results):.1f}°")
        print(f"   Average new error: {total_new_error / len(results):.1f}°")
        print(f"   Average improvement: {avg_improvement:.1f}°")

        improvements = sum(1 for r in results if r['improvement'] > 10)
        print(f"   Cases with >10° improvement: {improvements}/{len(results)}")

def test_on_larger_sample():
    """Test fixed algorithm on larger sample"""

    print(f"\n🧪 TESTING ON LARGER SAMPLE")
    print("=" * 50)

    # Read sample files
    with open('/tmp/sample_files.txt', 'r') as f:
        sample_files = [line.strip() for line in f.readlines()]

    manager = FlightManager()
    successes = 0
    improvements = 0
    total_old_error = 0
    total_new_error = 0

    for i, filepath in enumerate(sample_files[:20], 1):  # Test first 20
        try:
            # Get gswoop result
            result = subprocess.run(['gswoop', '-i', filepath],
                                  capture_output=True, text=True, timeout=30)
            gswoop_rotation = parse_gswoop_rotation(result.stdout)
            if not gswoop_rotation:
                continue

            # Analyze with both algorithms
            df, metadata = manager.read_flysight_file(filepath)
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Compare algorithms
            old_rotation, _, _, _ = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)
            new_rotation, _, _, _ = fixed_rotation_detection(df, flare_idx, max_gspeed_idx)

            old_diff = abs(old_rotation - gswoop_rotation)
            new_diff = abs(new_rotation - gswoop_rotation)

            if new_diff < old_diff:
                improvements += 1

            total_old_error += old_diff
            total_new_error += new_diff
            successes += 1

            if i % 5 == 0:
                print(f"   Processed: {i}, Improvements: {improvements}/{successes}")

        except Exception:
            continue

    if successes > 0:
        print(f"\n📈 LARGER SAMPLE RESULTS:")
        print(f"   Files processed: {successes}")
        print(f"   Improvements: {improvements} ({improvements/successes*100:.1f}%)")
        print(f"   Average error reduction: {(total_old_error - total_new_error)/successes:.1f}°")

if __name__ == "__main__":
    test_fixed_algorithm()
    test_on_larger_sample()