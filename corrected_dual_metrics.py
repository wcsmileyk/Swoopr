#!/usr/bin/env python3
"""
Corrected dual rotation metrics implementation
Based on discovery that gswoop measures turn start to rollout end, not to max vspeed
"""

import os
import sys
import django
import subprocess
import re
import numpy as np

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def parse_gswoop_altitudes(output):
    """Parse all altitude markers from gswoop output"""
    altitudes = {}

    patterns = {
        'turn_start': r'initiated turn:\s+(\d+)\s+ft AGL',
        'max_vspeed': r'max vertical speed:\s+(\d+)\s+ft AGL',
        'rollout_start': r'started rollout:\s+(\d+)\s+ft AGL',
        'rollout_end': r'finished rollout:\s+(\d+)\s+ft AGL',
        'max_total_speed': r'max total speed:\s+(\d+)\s+ft AGL',
        'max_horizontal_speed': r'max horizontal speed:\s+(\d+)\s+ft AGL',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            altitudes[key] = float(match.group(1))

    # Also get the rotation value
    rotation_match = re.search(r'degrees of rotation:\s+(\d+)\s+deg\s+\((\w+)-hand\)', output)
    if rotation_match:
        degrees = float(rotation_match.group(1))
        direction = rotation_match.group(2)
        altitudes['gswoop_rotation'] = degrees if direction == 'right' else -degrees

    return altitudes

def find_point_by_altitude(df, target_alt_ft, tolerance_ft=15):
    """Find GPS point closest to target altitude with tolerance"""
    target_alt_m = target_alt_ft * 0.3048
    altitude_diffs = np.abs(df['AGL'] - target_alt_m)
    closest_idx = altitude_diffs.idxmin()
    closest_alt_ft = df.loc[closest_idx, 'AGL'] / 0.3048
    diff_ft = abs(closest_alt_ft - target_alt_ft)

    if diff_ft <= tolerance_ft:
        return closest_idx, closest_alt_ft, diff_ft
    else:
        return None, closest_alt_ft, diff_ft

def calculate_corrected_dual_metrics(filepath):
    """Calculate both full swoop and corrected gswoop-style rotation metrics"""

    print(f"🔄 CORRECTED DUAL METRICS: {os.path.basename(filepath)}")
    print("=" * 60)

    # Get gswoop analysis
    result = subprocess.run(['gswoop', '-i', filepath],
                          capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        print("❌ gswoop failed")
        return None

    gswoop_altitudes = parse_gswoop_altitudes(result.stdout)

    # Load flight data
    manager = FlightManager()
    df, metadata = manager.read_flysight_file(filepath)

    print("📊 gswoop Altitude Markers:")
    for key, alt in gswoop_altitudes.items():
        if key != 'gswoop_rotation':
            print(f"   {key}: {alt} ft AGL")
    print(f"   Published rotation: {gswoop_altitudes.get('gswoop_rotation', 'N/A')}°")

    # Find our algorithm's boundaries
    landing_idx = manager.get_landing(df)
    try:
        flare_idx = manager.find_flare(df, landing_idx)
        flare_method = "traditional"
    except:
        flare_idx = manager.find_turn_start_fallback(df, landing_idx)
        flare_method = "fallback"

    max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

    results = {}

    # 1. FULL SWOOP ROTATION (our existing comprehensive method)
    full_rotation, intended_turn, confidence, method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

    results['full_swoop'] = {
        'rotation': full_rotation,
        'intended_turn': intended_turn,
        'confidence': confidence,
        'method': method,
        'start_alt': df.iloc[flare_idx]['AGL'] / 0.3048,
        'end_alt': df.iloc[max_gspeed_idx]['AGL'] / 0.3048,
        'duration': (max_gspeed_idx - flare_idx) * 0.2,
        'description': 'Full swoop from flare initiation to max ground speed'
    }

    # 2. GSWOOP-STYLE ROTATION (turn initiation to rollout end)
    if 'turn_start' in gswoop_altitudes and 'rollout_end' in gswoop_altitudes:

        # Find GPS points matching gswoop's altitude markers
        turn_start_idx, start_alt_actual, start_diff = find_point_by_altitude(df, gswoop_altitudes['turn_start'])
        rollout_end_idx, end_alt_actual, end_diff = find_point_by_altitude(df, gswoop_altitudes['rollout_end'])

        if turn_start_idx is not None and rollout_end_idx is not None:
            print(f"\n🎯 gswoop-style Boundaries Found:")
            print(f"   Turn start: idx {turn_start_idx}, {start_alt_actual:.1f}ft (±{start_diff:.1f}ft)")
            print(f"   Rollout end: idx {rollout_end_idx}, {end_alt_actual:.1f}ft (±{end_diff:.1f}ft)")

            if turn_start_idx < rollout_end_idx:
                # Calculate rotation over gswoop's segment
                turn_segment = df.iloc[turn_start_idx:rollout_end_idx+1]
                headings = turn_segment['heading'].values

                if len(headings) >= 2:
                    start_heading = headings[0]
                    end_heading = headings[-1]

                    # Calculate net rotation (like gswoop appears to do)
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

                    # Calculate confidence based on match with gswoop
                    gswoop_rotation = gswoop_altitudes.get('gswoop_rotation', 0)
                    diff_from_gswoop = abs(net_rotation - gswoop_rotation)
                    confidence = max(0.1, 1.0 - (diff_from_gswoop / 180))

                    results['gswoop_style'] = {
                        'rotation': net_rotation,
                        'intended_turn': intended_turn,
                        'confidence': confidence,
                        'method': 'gswoop_turn_to_rollout',
                        'start_alt': start_alt_actual,
                        'end_alt': end_alt_actual,
                        'duration': len(headings) * 0.2,
                        'description': 'gswoop-style: turn initiation to rollout end',
                        'gswoop_published': gswoop_rotation,
                        'difference_from_gswoop': diff_from_gswoop
                    }

                    print(f"   Start heading: {start_heading:.1f}°")
                    print(f"   End heading: {end_heading:.1f}°")
                    print(f"   Calculated rotation: {net_rotation:.1f}°")
                    print(f"   gswoop published: {gswoop_rotation:.1f}°")
                    print(f"   Difference: {diff_from_gswoop:.1f}°")

                    if diff_from_gswoop < 10:
                        print(f"   ✅ EXCELLENT MATCH!")
                    elif diff_from_gswoop < 30:
                        print(f"   ⚡ GOOD MATCH")
                    else:
                        print(f"   ⚠️  Still needs refinement")

    # 3. DISPLAY COMPARISON
    print(f"\n📊 DUAL METRICS COMPARISON:")

    if 'full_swoop' in results:
        fs = results['full_swoop']
        print(f"   🔄 Full Swoop:")
        print(f"      Rotation: {fs['rotation']:.1f}° → {fs['intended_turn']}°")
        print(f"      Segment: {fs['start_alt']:.0f}ft to {fs['end_alt']:.0f}ft ({fs['duration']:.1f}s)")
        print(f"      Confidence: {fs['confidence']:.2f} ({fs['method']})")

    if 'gswoop_style' in results:
        gs = results['gswoop_style']
        print(f"   📊 gswoop-style:")
        print(f"      Rotation: {gs['rotation']:.1f}° → {gs['intended_turn']}°")
        print(f"      Segment: {gs['start_alt']:.0f}ft to {gs['end_alt']:.0f}ft ({gs['duration']:.1f}s)")
        print(f"      Confidence: {gs['confidence']:.2f} (±{gs['difference_from_gswoop']:.1f}° from gswoop)")

    return results

def test_corrected_algorithm():
    """Test the corrected algorithm on our problem cases"""

    print("🧪 TESTING CORRECTED DUAL METRICS ALGORITHM")
    print("=" * 70)

    test_cases = [
        "~/FlySight/Training/25-02-20/25-02-20-sw1.csv",  # Expert: left 270
        "~/FlySight/Training/25-03-27/25-03-27-sw4.csv",  # Expert: left 270
        "~/FlySight/Training/24-10-12/24-10-12-sw3.csv",  # Known good case
    ]

    successes = 0

    for filepath in test_cases:
        filepath = os.path.expanduser(filepath)

        results = calculate_corrected_dual_metrics(filepath)

        if results and 'gswoop_style' in results:
            diff = results['gswoop_style']['difference_from_gswoop']
            if diff < 20:
                successes += 1
                print(f"   ✅ SUCCESS: Close match with gswoop")
            else:
                print(f"   ⚠️  Still working on this case")

        print("\n" + "="*70 + "\n")

    print(f"🎯 SUMMARY: {successes}/{len(test_cases)} cases successfully aligned with gswoop")

    if successes >= len(test_cases) * 0.7:
        print("✅ ALGORITHM READY: Good alignment with gswoop achieved!")
        print("📋 NEXT STEPS:")
        print("   - Integrate dual metrics into FlightManager")
        print("   - Add database fields for gswoop-style rotation")
        print("   - Train ML on clean aligned data")
    else:
        print("⚠️  NEEDS MORE WORK: Algorithm alignment needs refinement")

if __name__ == "__main__":
    test_corrected_algorithm()