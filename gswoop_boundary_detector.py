#!/usr/bin/env python3
"""
Reverse engineer gswoop's turn boundary detection using published altitude markers
Implement dual rotation metrics: Full Swoop vs Turn Segment
"""

import os
import sys
import django
import subprocess
import re
import numpy as np
import pandas as pd

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def parse_gswoop_boundaries(output):
    """Parse gswoop output to extract precise boundary markers"""
    boundaries = {}

    patterns = {
        'turn_start_alt': r'initiated turn:\s+(\d+)\s+ft AGL',
        'turn_start_back': r'initiated turn:\s+\d+\s+ft AGL,\s+(\d+)\s+ft back',
        'turn_start_offset': r'initiated turn:.*,\s+(\-?\d+)\s+ft offset',
        'max_vspeed_alt': r'max vertical speed:\s+(\d+)\s+ft AGL',
        'max_vspeed_back': r'max vertical speed:.*,\s+(\d+)\s+ft back',
        'max_vspeed_offset': r'max vertical speed:.*,\s+(\-?\d+)\s+ft offset',
        'max_vspeed_mph': r'max vertical speed:.*\((\d+\.\d+)\s+mph\)',
        'rollout_start_alt': r'started rollout:\s+(\d+)\s+ft AGL',
        'rollout_start_back': r'started rollout:.*,\s+(\d+)\s+ft back',
        'rollout_end_alt': r'finished rollout:\s+(\d+)\s+ft AGL',
        'rotation_deg': r'degrees of rotation:\s+(\d+)\s+deg',
        'rotation_dir': r'degrees of rotation:.*\((\w+)-hand\)',
        'turn_time': r'time to execute turn:\s+(\d+\.\d+)\s+sec',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                boundaries[key] = float(match.group(1))
            except:
                boundaries[key] = match.group(1)

    return boundaries

def find_gps_point_by_altitude(df, target_alt_ft, tolerance_ft=10):
    """Find GPS point closest to target altitude AGL"""
    # Convert target altitude to meters for comparison
    target_alt_m = target_alt_ft * 0.3048

    # Find points within tolerance
    altitude_diffs = np.abs(df['AGL'] - target_alt_m)
    closest_idx = altitude_diffs.idxmin()
    closest_alt_m = df.loc[closest_idx, 'AGL']
    closest_alt_ft = closest_alt_m / 0.3048

    diff_ft = abs(closest_alt_ft - target_alt_ft)

    if diff_ft <= tolerance_ft:
        return closest_idx, closest_alt_ft, diff_ft
    else:
        return None, closest_alt_ft, diff_ft

def reverse_engineer_gswoop_boundaries(filepath):
    """Reverse engineer gswoop's boundary detection using altitude markers"""

    print(f"🔍 Reverse Engineering: {os.path.basename(filepath)}")
    print("=" * 60)

    # Get gswoop analysis
    result = subprocess.run(['gswoop', '-i', filepath],
                          capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print("❌ gswoop failed")
        return None

    boundaries = parse_gswoop_boundaries(result.stdout)

    # Load flight data
    manager = FlightManager()
    df, metadata = manager.read_flysight_file(filepath)

    print("📊 gswoop Boundary Markers:")
    for key, value in boundaries.items():
        if 'alt' in key:
            print(f"   {key}: {value} ft AGL")

    # Find corresponding GPS points using altitude matching
    gswoop_indices = {}

    # Find turn start point
    if 'turn_start_alt' in boundaries:
        idx, actual_alt, diff = find_gps_point_by_altitude(df, boundaries['turn_start_alt'])
        if idx is not None:
            gswoop_indices['turn_start'] = idx
            print(f"   ✅ Turn start found: idx {idx}, {actual_alt:.1f}ft (diff: {diff:.1f}ft)")
        else:
            print(f"   ❌ Turn start not found within tolerance: closest {actual_alt:.1f}ft (diff: {diff:.1f}ft)")

    # Find max vspeed point
    if 'max_vspeed_alt' in boundaries:
        idx, actual_alt, diff = find_gps_point_by_altitude(df, boundaries['max_vspeed_alt'])
        if idx is not None:
            gswoop_indices['max_vspeed'] = idx
            print(f"   ✅ Max vspeed found: idx {idx}, {actual_alt:.1f}ft (diff: {diff:.1f}ft)")
        else:
            print(f"   ❌ Max vspeed not found within tolerance: closest {actual_alt:.1f}ft (diff: {diff:.1f}ft)")

    # Find rollout start if available
    if 'rollout_start_alt' in boundaries:
        idx, actual_alt, diff = find_gps_point_by_altitude(df, boundaries['rollout_start_alt'])
        if idx is not None:
            gswoop_indices['rollout_start'] = idx
            print(f"   ✅ Rollout start found: idx {idx}, {actual_alt:.1f}ft (diff: {diff:.1f}ft)")

    # Calculate gswoop-style rotation if we found the boundaries
    if 'turn_start' in gswoop_indices and 'max_vspeed' in gswoop_indices:
        gswoop_turn_segment = df.iloc[gswoop_indices['turn_start']:gswoop_indices['max_vspeed']+1]

        if len(gswoop_turn_segment) >= 2:
            headings = gswoop_turn_segment['heading'].values
            start_heading = headings[0]
            end_heading = headings[-1]

            # Calculate net rotation
            net_rotation = end_heading - start_heading
            while net_rotation > 180:
                net_rotation -= 360
            while net_rotation < -180:
                net_rotation += 360

            gswoop_rotation = boundaries.get('rotation_deg', 0)
            if boundaries.get('rotation_dir') == 'left':
                gswoop_rotation = -gswoop_rotation

            print(f"\n📐 Turn Segment Analysis (gswoop boundaries):")
            print(f"   Start heading: {start_heading:.1f}°")
            print(f"   End heading: {end_heading:.1f}°")
            print(f"   Our calculated net rotation: {net_rotation:.1f}°")
            print(f"   gswoop published rotation: {gswoop_rotation:.1f}°")
            print(f"   Difference: {abs(net_rotation - gswoop_rotation):.1f}°")
            print(f"   Turn segment points: {len(gswoop_turn_segment)}")
            print(f"   Turn segment duration: {len(gswoop_turn_segment) * 0.2:.1f} sec")

            # Check if net rotation matches gswoop closely
            if abs(net_rotation - gswoop_rotation) < 20:
                print(f"   ✅ CLOSE MATCH: gswoop likely uses net heading change")
            else:
                print(f"   ⚠️  DIFFERENCE: gswoop may use different calculation method")

    return {
        'boundaries': boundaries,
        'indices': gswoop_indices,
        'df': df
    }

def implement_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, gswoop_turn_start=None, gswoop_turn_end=None):
    """Calculate both full swoop and turn segment rotation metrics"""

    results = {}

    # 1. Full Swoop Rotation (our existing method)
    full_turn_data = df[flare_idx:max_gspeed_idx+1]
    full_headings = full_turn_data['heading'].values

    if len(full_headings) >= 2:
        # Use our existing improved algorithm
        manager = FlightManager()
        full_rotation, intended_turn, confidence, method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

        results['full_swoop'] = {
            'rotation': full_rotation,
            'intended_turn': intended_turn,
            'confidence': confidence,
            'method': method,
            'duration': len(full_headings) * 0.2,
            'segment': 'flare_to_max_gspeed'
        }

    # 2. Turn Segment Rotation (gswoop-style)
    if gswoop_turn_start is not None and gswoop_turn_end is not None:
        turn_segment_data = df.iloc[gswoop_turn_start:gswoop_turn_end+1]
        segment_headings = turn_segment_data['heading'].values

        if len(segment_headings) >= 2:
            # Simple net rotation calculation (like gswoop appears to use)
            start_heading = segment_headings[0]
            end_heading = segment_headings[-1]

            net_rotation = end_heading - start_heading
            while net_rotation > 180:
                net_rotation -= 360
            while net_rotation < -180:
                net_rotation += 360

            # Classify intended turn
            abs_rotation = abs(net_rotation)
            if abs_rotation < 150:
                intended_turn = 90
            elif abs_rotation < 350:
                intended_turn = 270
            elif abs_rotation < 550:
                intended_turn = 450
            else:
                intended_turn = 630

            # Simple confidence based on how close to standard turn
            distance_to_standard = min([abs(abs_rotation - std) for std in [90, 270, 450, 630]])
            confidence = max(0.3, 1.0 - (distance_to_standard / 90))

            results['turn_segment'] = {
                'rotation': net_rotation,
                'intended_turn': intended_turn,
                'confidence': confidence,
                'method': 'gswoop_style_net',
                'duration': len(segment_headings) * 0.2,
                'segment': 'turn_initiation_to_max_vspeed'
            }

    return results

def test_dual_metrics():
    """Test dual rotation metrics on sample files"""

    print("🧪 TESTING DUAL ROTATION METRICS")
    print("=" * 60)

    test_files = [
        "~/FlySight/Training/25-02-20/25-02-20-sw1.csv",
        "~/FlySight/Training/25-03-27/25-03-27-sw4.csv",
        "~/FlySight/Training/24-10-12/24-10-12-sw3.csv",
    ]

    for filepath in test_files:
        filepath = os.path.expanduser(filepath)

        # Reverse engineer gswoop boundaries
        analysis = reverse_engineer_gswoop_boundaries(filepath)

        if analysis and 'turn_start' in analysis['indices']:
            # Get our algorithm's boundaries
            manager = FlightManager()
            df = analysis['df']
            landing_idx = manager.get_landing(df)

            try:
                flare_idx = manager.find_flare(df, landing_idx)
            except:
                flare_idx = manager.find_turn_start_fallback(df, landing_idx)

            max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

            # Calculate dual metrics
            gswoop_start = analysis['indices'].get('turn_start')
            gswoop_end = analysis['indices'].get('max_vspeed', analysis['indices'].get('rollout_start'))

            dual_metrics = implement_dual_rotation_metrics(
                df, flare_idx, max_gspeed_idx, gswoop_start, gswoop_end
            )

            print(f"\n📊 DUAL METRICS SUMMARY:")

            if 'full_swoop' in dual_metrics:
                fs = dual_metrics['full_swoop']
                print(f"   Full Swoop: {fs['rotation']:.1f}° → {fs['intended_turn']}° (conf: {fs['confidence']:.2f}, {fs['duration']:.1f}s)")

            if 'turn_segment' in dual_metrics:
                ts = dual_metrics['turn_segment']
                print(f"   Turn Segment: {ts['rotation']:.1f}° → {ts['intended_turn']}° (conf: {ts['confidence']:.2f}, {ts['duration']:.1f}s)")

                # Compare with gswoop published value
                gswoop_rotation = analysis['boundaries'].get('rotation_deg', 0)
                if analysis['boundaries'].get('rotation_dir') == 'left':
                    gswoop_rotation = -gswoop_rotation

                diff = abs(ts['rotation'] - gswoop_rotation)
                print(f"   gswoop Published: {gswoop_rotation:.1f}° (diff: {diff:.1f}°)")

                if diff < 10:
                    print(f"   ✅ EXCELLENT MATCH with gswoop!")
                elif diff < 30:
                    print(f"   ⚡ GOOD MATCH with gswoop")
                else:
                    print(f"   ⚠️  Still different from gswoop")

        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_dual_metrics()