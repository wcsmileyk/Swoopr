#!/usr/bin/env python3
"""
Batch analysis of rotation disagreements for expert review
Shows top disagreements with detailed analysis
"""

import os
import sys
import django
import subprocess
import re
import json
from pathlib import Path

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

def analyze_file_detailed(filepath, manager):
    """Detailed analysis of a single file"""
    # Get gswoop result
    result = subprocess.run(['gswoop', '-i', filepath],
                          capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None

    gswoop_rotation = parse_gswoop_rotation(result.stdout)
    if gswoop_rotation is None:
        return None

    # Analyze with our algorithm
    try:
        df, metadata = manager.read_flysight_file(filepath)
        landing_idx = manager.get_landing(df)

        try:
            flare_idx = manager.find_flare(df, landing_idx)
            flare_method = "traditional"
        except:
            flare_idx = manager.find_turn_start_fallback(df, landing_idx)
            flare_method = "fallback"

        max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)
        our_rotation, intended_turn, confidence, method = manager.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

        # Detailed turn analysis
        turn_data = df[flare_idx:max_gspeed_idx+1]
        headings = turn_data['heading'].values

        # Calculate heading statistics
        net_change = headings[-1] - headings[0]
        while net_change > 180:
            net_change -= 360
        while net_change < -180:
            net_change += 360

        # Calculate total heading change
        total_change = 0
        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            total_change += abs(diff)

        return {
            'filepath': filepath,
            'filename': Path(filepath).name,
            'gswoop_rotation': gswoop_rotation,
            'our_rotation': our_rotation,
            'our_confidence': confidence,
            'our_method': method,
            'our_intended': intended_turn,
            'flare_method': flare_method,
            'difference': abs(our_rotation - gswoop_rotation),
            'turn_duration': len(turn_data) * 0.2,
            'entry_altitude': df.iloc[flare_idx]['AGL'],
            'turn_points': len(turn_data),
            'heading_start': headings[0],
            'heading_end': headings[-1],
            'net_heading_change': net_change,
            'total_heading_change': total_change,
            'gswoop_full_output': result.stdout
        }
    except Exception as e:
        return None

def analyze_top_disagreements():
    """Analyze and display top disagreements"""

    # Get sample files
    with open('/tmp/sample_files.txt', 'r') as f:
        sample_files = [line.strip() for line in f.readlines()]

    print("🔍 DETAILED ANALYSIS OF ROTATION DISAGREEMENTS")
    print("=" * 80)

    manager = FlightManager()
    all_cases = []

    for filepath in sample_files:
        result = analyze_file_detailed(filepath, manager)
        if result and result['difference'] > 50:  # Only significant disagreements
            all_cases.append(result)

    # Sort by difference magnitude
    all_cases.sort(key=lambda x: x['difference'], reverse=True)

    print(f"Found {len(all_cases)} significant disagreements (>50°)")
    print("\n" + "="*80)

    # Show top 10 cases for expert review
    for i, case in enumerate(all_cases[:10], 1):
        print(f"\n📁 CASE {i}: {case['filename']}")
        print(f"{'='*60}")

        print(f"🔄 ROTATION COMPARISON:")
        print(f"   Our algorithm: {case['our_rotation']:.1f}° (intended: {case['our_intended']}°, confidence: {case['our_confidence']:.2f})")
        print(f"   gswoop:        {case['gswoop_rotation']:.1f}°")
        print(f"   Difference:    {case['difference']:.1f}°")

        print(f"\n📐 HEADING ANALYSIS:")
        print(f"   Start heading: {case['heading_start']:.1f}°")
        print(f"   End heading:   {case['heading_end']:.1f}°")
        print(f"   Net change:    {case['net_heading_change']:.1f}°")
        print(f"   Total change:  {case['total_heading_change']:.1f}°")

        print(f"\n📊 FLIGHT DETAILS:")
        print(f"   Turn duration:   {case['turn_duration']:.1f} seconds")
        print(f"   Entry altitude:  {case['entry_altitude']:.0f} ft AGL")
        print(f"   GPS points:      {case['turn_points']}")
        print(f"   Flare method:    {case['flare_method']}")
        print(f"   Our method:      {case['our_method']}")

        print(f"\n🔍 POTENTIAL ISSUES:")

        # Analyze potential causes
        if abs(case['difference'] - 360) < 50:
            print(f"   ⚠️  ~360° difference suggests direction flip issue")

        if case['our_intended'] > 450 and abs(case['gswoop_rotation']) < 360:
            print(f"   ⚠️  Our algorithm detecting multiple rotations vs gswoop single rotation")

        if case['flare_method'] == 'fallback':
            print(f"   ⚠️  Using fallback flare detection - may affect turn boundaries")

        if case['our_confidence'] < 0.5:
            print(f"   ⚠️  Low confidence in our algorithm")

        net_vs_total_ratio = abs(case['net_heading_change']) / case['total_heading_change']
        if net_vs_total_ratio < 0.5:
            print(f"   ⚠️  Significant oscillation detected (net/total ratio: {net_vs_total_ratio:.2f})")

        print(f"\n💭 EXPERT QUESTION:")
        print(f"   Looking at this data, which rotation value seems more accurate?")
        print(f"   Consider: flight path, turn characteristics, and typical swoop patterns")

    # Save detailed analysis
    output_file = '/home/smiley/PycharmProjects/Swoopr/disagreement_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(all_cases, f, indent=2)

    print(f"\n💾 Detailed analysis saved to: {output_file}")
    print(f"\n📋 SUMMARY:")
    print(f"   Total disagreements analyzed: {len(all_cases)}")
    print(f"   Top 10 cases shown above for expert review")
    print(f"   Average difference: {sum(c['difference'] for c in all_cases) / len(all_cases):.1f}°")

    # Pattern analysis
    direction_flips = sum(1 for c in all_cases if abs(c['difference'] - 360) < 50)
    multi_rotation_issues = sum(1 for c in all_cases if c['our_intended'] > 450 and abs(c['gswoop_rotation']) < 360)
    fallback_issues = sum(1 for c in all_cases if c['flare_method'] == 'fallback')

    print(f"\n🔍 PATTERN ANALYSIS:")
    print(f"   Likely direction flips (~360° diff): {direction_flips}")
    print(f"   Multi-rotation disagreements: {multi_rotation_issues}")
    print(f"   Cases using fallback flare detection: {fallback_issues}")

    print(f"\n💡 RECOMMENDATIONS:")
    if direction_flips > len(all_cases) * 0.3:
        print(f"   - Focus on fixing direction interpretation logic")
    if multi_rotation_issues > len(all_cases) * 0.2:
        print(f"   - Review full rotation detection algorithm")
    if fallback_issues > len(all_cases) * 0.3:
        print(f"   - Improve flare detection to reduce fallback usage")

if __name__ == "__main__":
    analyze_top_disagreements()