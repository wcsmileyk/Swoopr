#!/usr/bin/env python3
"""
Comprehensive analysis of turn rotation patterns in sample flight data
"""

import os
import sys
import django
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt  # Not needed for this analysis
from pathlib import Path

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager, compute_heading

def analyze_turn_rotation(df, flare_idx, max_gspeed_idx, filename):
    """Analyze turn rotation with multiple methods"""

    print(f"\n{'='*60}")
    print(f"📁 ANALYZING: {filename}")
    print(f"{'='*60}")

    # Basic turn window info
    turn_duration = (max_gspeed_idx - flare_idx) * 0.2  # 5Hz data
    print(f"🕐 Turn duration: {turn_duration:.1f}s ({max_gspeed_idx - flare_idx} records)")

    # Extract turn data
    turn_data = df[flare_idx:max_gspeed_idx+1].copy()
    headings = turn_data['heading'].values
    times = turn_data['t_s'].values

    print(f"📐 Start heading: {headings[0]:.1f}°")
    print(f"📐 End heading: {headings[-1]:.1f}°")

    # Method 1: Current FlightManager algorithm
    manager = FlightManager()
    current_rotation = manager.get_rotation(df, flare_idx, max_gspeed_idx)
    print(f"🔄 Current algorithm: {current_rotation:.1f}°")

    # Method 2: Simple cumulative angular distance
    def calculate_cumulative_turn(headings):
        total_turn = 0
        direction_changes = 0
        last_direction = None

        for i in range(1, len(headings)):
            # Calculate angular difference
            diff = headings[i] - headings[i-1]

            # Normalize to [-180, 180]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            total_turn += abs(diff)

            # Track direction changes
            current_direction = 1 if diff > 0 else -1 if diff < 0 else 0
            if current_direction != 0 and last_direction is not None and current_direction != last_direction:
                direction_changes += 1
            if current_direction != 0:
                last_direction = current_direction

        return total_turn, direction_changes

    cumulative_turn, direction_changes = calculate_cumulative_turn(headings)
    print(f"🔄 Cumulative turn: {cumulative_turn:.1f}° ({direction_changes} direction changes)")

    # Method 3: Net angular change + full rotations
    def calculate_net_plus_rotations(headings):
        # Calculate net change
        net_change = headings[-1] - headings[0]
        while net_change > 180:
            net_change -= 360
        while net_change < -180:
            net_change += 360

        # Estimate full rotations by looking at cumulative distance
        cumulative_distance = 0
        prev_heading = headings[0]

        for heading in headings[1:]:
            diff = heading - prev_heading
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            cumulative_distance += abs(diff)
            prev_heading = heading

        # Estimate full rotations
        estimated_full_rotations = max(0, int((cumulative_distance - abs(net_change)) / 360))
        total_rotation = abs(net_change) + (estimated_full_rotations * 360)

        return total_rotation, net_change, estimated_full_rotations

    net_rotation, net_change, full_rotations = calculate_net_plus_rotations(headings)
    print(f"🔄 Net + rotations: {net_rotation:.1f}° (net: {net_change:.1f}°, full: {full_rotations})")

    # Method 4: Turn rate analysis
    def analyze_turn_rates(headings, times):
        rates = []
        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            time_diff = times[i] - times[i-1]
            if time_diff > 0:
                rate = abs(diff) / time_diff
                rates.append(rate)

        return rates

    turn_rates = analyze_turn_rates(headings, times)
    if turn_rates:
        avg_rate = np.mean(turn_rates)
        max_rate = np.max(turn_rates)
        print(f"🏃 Turn rates: avg={avg_rate:.1f}°/s, max={max_rate:.1f}°/s")

    # Method 5: Intended turn inference
    def infer_intended_turn(calculated_rotation):
        """Infer intended turn from standard increments: 90, 270, 450, 630, 810, 990"""
        standard_turns = [90, 270, 450, 630, 810, 990]

        # Find closest standard turn
        distances = [abs(calculated_rotation - std) for std in standard_turns]
        closest_idx = np.argmin(distances)
        closest_turn = standard_turns[closest_idx]
        distance = distances[closest_idx]

        # Only accept if within reasonable tolerance (±60°)
        if distance <= 60:
            confidence = max(0, (60 - distance) / 60)  # Confidence based on distance
            return closest_turn, confidence
        else:
            return None, 0

    # Test intended turn inference on different rotation estimates
    methods = [
        ("Current", current_rotation),
        ("Cumulative", cumulative_turn),
        ("Net+Rotations", net_rotation)
    ]

    print(f"\n🎯 INTENDED TURN INFERENCE:")
    for method_name, rotation in methods:
        intended, confidence = infer_intended_turn(abs(rotation))
        if intended:
            print(f"   {method_name}: {intended}° (confidence: {confidence:.2f})")
        else:
            print(f"   {method_name}: Non-standard turn")

    return {
        'filename': filename,
        'current_rotation': current_rotation,
        'cumulative_turn': cumulative_turn,
        'net_rotation': net_rotation,
        'avg_turn_rate': np.mean(turn_rates) if turn_rates else 0,
        'max_turn_rate': np.max(turn_rates) if turn_rates else 0,
        'turn_duration': turn_duration,
        'direction_changes': direction_changes
    }

def analyze_all_samples():
    """Analyze all sample files"""

    sample_dir = Path('/home/smiley/PycharmProjects/Swoopr/sample_tracks')
    results = []

    print("🔍 ANALYZING SAMPLE FLIGHT DATA FOR TURN PATTERNS")
    print("=" * 80)

    for csv_file in sample_dir.glob('*.csv'):
        if csv_file.name == 'gps_00545.csv':
            continue  # Skip the different format file for now

        try:
            # Load and process file
            manager = FlightManager()
            df, metadata = manager.read_flysight_file(str(csv_file))

            # Find turn points
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
                flare_method = "traditional"
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)
                flare_method = "turn_detection"

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Analyze this turn
            result = analyze_turn_rotation(df, flare_idx, max_gspeed_idx, csv_file.name)
            results.append(result)

        except Exception as e:
            print(f"❌ Error analyzing {csv_file.name}: {e}")

    # Summary analysis
    if results:
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY OF {len(results)} FLIGHTS")
        print(f"{'='*80}")

        # Group by rotation ranges
        rotation_groups = {
            '90° range (60-120°)': [],
            '270° range (240-300°)': [],
            '450° range (420-480°)': [],
            '630° range (600-660°)': [],
            'Other': []
        }

        for result in results:
            rotation = abs(result['current_rotation'])
            if 60 <= rotation <= 120:
                rotation_groups['90° range (60-120°)'].append(result)
            elif 240 <= rotation <= 300:
                rotation_groups['270° range (240-300°)'].append(result)
            elif 420 <= rotation <= 480:
                rotation_groups['450° range (420-480°)'].append(result)
            elif 600 <= rotation <= 660:
                rotation_groups['630° range (600-660°)'].append(result)
            else:
                rotation_groups['Other'].append(result)

        for group_name, group_results in rotation_groups.items():
            if group_results:
                print(f"\n🎯 {group_name}: {len(group_results)} flights")
                avg_current = np.mean([r['current_rotation'] for r in group_results])
                avg_cumulative = np.mean([r['cumulative_turn'] for r in group_results])
                avg_rate = np.mean([r['avg_turn_rate'] for r in group_results])

                print(f"   Avg current rotation: {avg_current:.1f}°")
                print(f"   Avg cumulative turn: {avg_cumulative:.1f}°")
                print(f"   Avg turn rate: {avg_rate:.1f}°/s")

                for r in group_results:
                    print(f"     {r['filename']}: {r['current_rotation']:.1f}°")

if __name__ == "__main__":
    analyze_all_samples()