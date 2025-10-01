#!/usr/bin/env python3
"""
Interactive rotation validation tool for training data curation
Highlights significant differences between our algorithm and gswoop for human review
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

def analyze_file(filepath, manager):
    """Analyze a single file with both methods"""
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

        # Calculate turn segment details
        turn_data = df[flare_idx:max_gspeed_idx+1]

        return {
            'filepath': filepath,
            'filename': Path(filepath).name,
            'gswoop_rotation': gswoop_rotation,
            'our_rotation': our_rotation,
            'our_confidence': confidence,
            'our_method': method,
            'flare_method': flare_method,
            'difference': abs(our_rotation - gswoop_rotation),
            'turn_duration': len(turn_data) * 0.2,
            'entry_altitude': df.iloc[flare_idx]['AGL'],
            'turn_points': len(turn_data)
        }
    except Exception as e:
        return None

def create_validation_session():
    """Create a validation session for significant disagreements"""

    # Get sample files (or use all training files)
    with open('/tmp/sample_files.txt', 'r') as f:
        sample_files = [line.strip() for line in f.readlines()]

    print("🔍 Analyzing files for significant disagreements...")
    print("=" * 60)

    manager = FlightManager()
    disagreements = []
    total_analyzed = 0

    for filepath in sample_files:
        result = analyze_file(filepath, manager)
        if result:
            total_analyzed += 1

            # Flag significant disagreements (>50° difference)
            if result['difference'] > 50:
                disagreements.append(result)

        if total_analyzed % 10 == 0:
            print(f"   Analyzed: {total_analyzed}, Disagreements: {len(disagreements)}")

    # Sort by difference magnitude
    disagreements.sort(key=lambda x: x['difference'], reverse=True)

    print(f"\n📊 Analysis Complete:")
    print(f"   Total files analyzed: {total_analyzed}")
    print(f"   Significant disagreements (>50°): {len(disagreements)}")
    print(f"   Agreement rate: {((total_analyzed - len(disagreements)) / total_analyzed * 100):.1f}%")

    return disagreements

def interactive_validation(disagreements):
    """Interactive validation of disagreements"""

    print(f"\n🎯 INTERACTIVE VALIDATION SESSION")
    print(f"{'='*60}")
    print(f"Found {len(disagreements)} significant disagreements to review")
    print(f"Commands: 'ours', 'gswoop', 'skip', 'quit', 'show' (show file details)")
    print()

    validation_results = []

    for i, case in enumerate(disagreements):
        print(f"\n📁 [{i+1}/{len(disagreements)}] {case['filename']}")
        print(f"   🔄 Our algorithm: {case['our_rotation']:.1f}° (confidence: {case['our_confidence']:.2f}, method: {case['our_method']})")
        print(f"   📊 gswoop:        {case['gswoop_rotation']:.1f}°")
        print(f"   ❗ Difference:     {case['difference']:.1f}°")
        print(f"   📈 Details: {case['turn_duration']:.1f}s turn, {case['entry_altitude']:.0f}ft entry, {case['turn_points']} GPS points")

        while True:
            choice = input(f"   Which is more accurate? (ours/gswoop/show/skip/quit): ").strip().lower()

            if choice == 'ours':
                validation_results.append({
                    **case,
                    'human_choice': 'ours',
                    'correct_rotation': case['our_rotation']
                })
                print(f"   ✅ Marked: Our algorithm is correct")
                break

            elif choice == 'gswoop':
                validation_results.append({
                    **case,
                    'human_choice': 'gswoop',
                    'correct_rotation': case['gswoop_rotation']
                })
                print(f"   ✅ Marked: gswoop is correct")
                break

            elif choice == 'show':
                # Show detailed file analysis
                print(f"\n   📋 Detailed Analysis for {case['filename']}:")
                print(f"      File: {case['filepath']}")
                print(f"      Flare detection: {case['flare_method']}")
                print(f"      Turn duration: {case['turn_duration']:.1f} seconds")
                print(f"      Entry altitude: {case['entry_altitude']:.0f} ft AGL")
                print(f"      GPS points in turn: {case['turn_points']}")
                print(f"      Our confidence: {case['our_confidence']:.2f}")
                print(f"      Our method: {case['our_method']}")
                print(f"      Rotation difference: {case['difference']:.1f}°")

                # Option to open file in viewer/editor
                view_choice = input(f"      Open file? (y/n): ").strip().lower()
                if view_choice == 'y':
                    os.system(f"head -20 '{case['filepath']}'")

            elif choice == 'skip':
                print(f"   ⏭️  Skipped")
                break

            elif choice == 'quit':
                print(f"   🛑 Validation session ended")
                return validation_results

            else:
                print(f"   ❌ Invalid choice. Use: ours, gswoop, show, skip, or quit")

    return validation_results

def save_validation_results(results):
    """Save validation results for training"""
    output_file = '/home/smiley/PycharmProjects/Swoopr/rotation_validation_results.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Validation results saved to: {output_file}")

    # Summary
    ours_correct = sum(1 for r in results if r['human_choice'] == 'ours')
    gswoop_correct = sum(1 for r in results if r['human_choice'] == 'gswoop')

    print(f"\n📊 Validation Summary:")
    print(f"   Total cases reviewed: {len(results)}")
    print(f"   Our algorithm correct: {ours_correct} ({ours_correct/len(results)*100:.1f}%)")
    print(f"   gswoop correct: {gswoop_correct} ({gswoop_correct/len(results)*100:.1f}%)")

    if ours_correct > gswoop_correct:
        print(f"   💡 Recommendation: Focus on improving gswoop alignment for these cases")
    elif gswoop_correct > ours_correct:
        print(f"   💡 Recommendation: Align our algorithm more closely with gswoop")
    else:
        print(f"   💡 Recommendation: Both methods have merit, use ML to learn the best approach")

if __name__ == "__main__":
    print("🎯 ROTATION VALIDATION TOOL")
    print("=" * 40)
    print("This tool finds significant disagreements between our algorithm and gswoop")
    print("for human expert validation to create high-quality training data.")
    print()

    # Step 1: Find disagreements
    disagreements = create_validation_session()

    if not disagreements:
        print("✅ No significant disagreements found! Our algorithm aligns well with gswoop.")
        sys.exit(0)

    # Step 2: Interactive validation
    proceed = input(f"\nProceed with interactive validation of {len(disagreements)} cases? (y/n): ")
    if proceed.lower() != 'y':
        print("Validation session cancelled.")
        sys.exit(0)

    validation_results = interactive_validation(disagreements)

    if validation_results:
        save_validation_results(validation_results)