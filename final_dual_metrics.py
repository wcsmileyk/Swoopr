#!/usr/bin/env python3
"""
Final dual rotation metrics with intelligent direction detection
Analyzes intermediate headings to determine actual turn direction
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

    # Get rotation with direction
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

def calculate_smart_rotation(headings):
    """
    Calculate rotation using intermediate headings to determine actual turn direction
    This handles the case where shortest angular distance doesn't reflect actual turn
    """
    if len(headings) < 2:
        return 0, 0

    start_heading = headings[0]
    end_heading = headings[-1]

    # Calculate cumulative turn by following the actual path
    total_left_turn = 0
    total_right_turn = 0

    for i in range(1, len(headings)):
        prev_heading = headings[i-1]
        curr_heading = headings[i]

        # Calculate change
        diff = curr_heading - prev_heading

        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360

        # Filter out GPS noise (large jumps)
        if abs(diff) <= 90:
            if diff < 0:
                total_left_turn += abs(diff)
            else:
                total_right_turn += abs(diff)

    # Determine dominant turn direction
    if total_left_turn > total_right_turn:
        # Left turn dominant
        net_rotation = -(total_left_turn - total_right_turn)
        total_rotation = total_left_turn + total_right_turn
        direction = "left"
    else:
        # Right turn dominant
        net_rotation = total_right_turn - total_left_turn
        total_rotation = total_left_turn + total_right_turn
        direction = "right"

    # For significant turns (>180°), add full rotations
    if total_rotation > 180:
        # Check if we've made approximately full rotations
        estimated_full_rotations = int(total_rotation / 360)

        if direction == "left":
            final_rotation = -(abs(net_rotation) + estimated_full_rotations * 360)
        else:
            final_rotation = abs(net_rotation) + estimated_full_rotations * 360
    else:
        final_rotation = net_rotation

    return final_rotation, direction

def calculate_final_dual_metrics(filepath):
    """Calculate both full swoop and intelligent gswoop-style rotation metrics"""

    print(f"🎯 FINAL DUAL METRICS: {os.path.basename(filepath)}")
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

    print("📊 gswoop Reference:")
    print(f"   Published rotation: {gswoop_altitudes.get('gswoop_rotation', 'N/A')}°")

    # Find our algorithm's boundaries
    landing_idx = manager.get_landing(df)
    try:
        flare_idx = manager.find_flare(df, landing_idx)
    except:
        flare_idx = manager.find_turn_start_fallback(df, landing_idx)

    max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

    results = {}

    # 1. FULL SWOOP ROTATION (our comprehensive method)
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

    # 2. INTELLIGENT GSWOOP-STYLE ROTATION
    if 'turn_start' in gswoop_altitudes and 'rollout_end' in gswoop_altitudes:

        # Find GPS points matching gswoop's altitude markers
        turn_start_idx, start_alt_actual, start_diff = find_point_by_altitude(df, gswoop_altitudes['turn_start'])
        rollout_end_idx, end_alt_actual, end_diff = find_point_by_altitude(df, gswoop_altitudes['rollout_end'])

        if turn_start_idx is not None and rollout_end_idx is not None and turn_start_idx < rollout_end_idx:

            # Get turn segment
            turn_segment = df.iloc[turn_start_idx:rollout_end_idx+1]
            headings = turn_segment['heading'].values

            if len(headings) >= 2:
                # Use intelligent rotation calculation
                smart_rotation, turn_direction = calculate_smart_rotation(headings)

                # Also calculate simple net for comparison
                start_heading = headings[0]
                end_heading = headings[-1]
                simple_net = end_heading - start_heading
                while simple_net > 180:
                    simple_net -= 360
                while simple_net < -180:
                    simple_net += 360

                # Classify intended turn
                abs_rotation = abs(smart_rotation)
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
                diff_from_gswoop = abs(smart_rotation - gswoop_rotation)
                confidence = max(0.1, 1.0 - (diff_from_gswoop / 180))

                results['gswoop_style'] = {
                    'rotation': smart_rotation,
                    'intended_turn': intended_turn,
                    'confidence': confidence,
                    'method': 'intelligent_path_following',
                    'start_alt': start_alt_actual,
                    'end_alt': end_alt_actual,
                    'duration': len(headings) * 0.2,
                    'description': 'Intelligent path-following rotation analysis',
                    'gswoop_published': gswoop_rotation,
                    'difference_from_gswoop': diff_from_gswoop,
                    'turn_direction': turn_direction,
                    'simple_net': simple_net
                }

                print(f"\n🧠 Intelligent Analysis:")
                print(f"   Start heading: {start_heading:.1f}°")
                print(f"   End heading: {end_heading:.1f}°")
                print(f"   Simple net: {simple_net:.1f}°")
                print(f"   Smart rotation: {smart_rotation:.1f}° ({turn_direction} turn)")
                print(f"   gswoop published: {gswoop_rotation:.1f}°")
                print(f"   Difference: {diff_from_gswoop:.1f}°")

                if diff_from_gswoop < 10:
                    print(f"   ✅ EXCELLENT MATCH!")
                elif diff_from_gswoop < 30:
                    print(f"   ⚡ GOOD MATCH")
                else:
                    print(f"   ⚠️  Needs more refinement")

    # 3. DISPLAY COMPARISON
    print(f"\n📊 FINAL DUAL METRICS:")

    if 'full_swoop' in results:
        fs = results['full_swoop']
        print(f"   🔄 Full Swoop: {fs['rotation']:.1f}° → {fs['intended_turn']}° ({fs['start_alt']:.0f}-{fs['end_alt']:.0f}ft, {fs['duration']:.1f}s)")

    if 'gswoop_style' in results:
        gs = results['gswoop_style']
        print(f"   🧠 gswoop-style: {gs['rotation']:.1f}° → {gs['intended_turn']}° ({gs['start_alt']:.0f}-{gs['end_alt']:.0f}ft, {gs['duration']:.1f}s)")
        print(f"      Accuracy: ±{gs['difference_from_gswoop']:.1f}° from gswoop (confidence: {gs['confidence']:.2f})")

    return results

def test_final_algorithm():
    """Test the final algorithm with intelligent direction detection"""

    print("🎯 TESTING FINAL INTELLIGENT DUAL METRICS")
    print("=" * 70)

    test_cases = [
        "~/FlySight/Training/25-02-20/25-02-20-sw1.csv",  # Expert: left 270
        "~/FlySight/Training/25-03-27/25-03-27-sw4.csv",  # Expert: left 270
        "~/FlySight/Training/24-10-12/24-10-12-sw3.csv",  # Known good case
    ]

    excellent_matches = 0
    good_matches = 0

    for filepath in test_cases:
        filepath = os.path.expanduser(filepath)

        results = calculate_final_dual_metrics(filepath)

        if results and 'gswoop_style' in results:
            diff = results['gswoop_style']['difference_from_gswoop']
            if diff < 10:
                excellent_matches += 1
                print(f"   ✅ EXCELLENT MATCH")
            elif diff < 30:
                good_matches += 1
                print(f"   ⚡ GOOD MATCH")
            else:
                print(f"   ⚠️  Still needs work")

        print("\n" + "="*70 + "\n")

    total_matches = excellent_matches + good_matches
    print(f"🎯 FINAL RESULTS:")
    print(f"   Excellent matches (<10°): {excellent_matches}/{len(test_cases)}")
    print(f"   Good matches (<30°): {good_matches}/{len(test_cases)}")
    print(f"   Total acceptable: {total_matches}/{len(test_cases)} ({total_matches/len(test_cases)*100:.1f}%)")

    if total_matches >= len(test_cases) * 0.8:
        print("\n🎉 SUCCESS! Algorithm ready for production integration!")
        print("📋 NEXT STEPS:")
        print("   ✅ Integrate into FlightManager")
        print("   ✅ Add database fields for dual metrics")
        print("   ✅ Train ML on clean aligned data")
        print("   ✅ Deploy dual rotation analysis system")
    else:
        print("\n⚠️  Needs additional refinement")

if __name__ == "__main__":
    test_final_algorithm()