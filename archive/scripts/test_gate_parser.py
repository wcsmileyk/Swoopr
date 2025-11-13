#!/usr/bin/env python
"""
Test script to parse gate files and verify the gate parsing system works correctly.
"""

import os
import sys
import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.utils.gate_parser import GateFileParser
from flights.utils.course_builder import CourseBuilder
import json


def test_gate_file(file_path, gate_type):
    """Test parsing a gate file"""
    print(f"\n{'='*80}")
    print(f"Testing {gate_type} gate file: {file_path}")
    print(f"{'='*80}\n")

    try:
        # Parse the gate file
        parser = GateFileParser(file_path, gate_type)
        gate_positions = parser.extract_gate_positions()

        print("✓ Successfully parsed gate file!")
        print(f"\nGate Positions:")
        print(json.dumps(gate_positions, indent=2))

        # Build course
        builder = CourseBuilder(gate_positions, gate_type)
        course_config = builder.build_course()

        print(f"\n✓ Successfully built course configuration!")
        print(f"\nCourse Type: {course_config['gate_type']}")
        print(f"Course Bearing: {course_config['course_bearing']:.1f}°")

        print(f"\nEntry Gates:")
        for gate in course_config['entry_gates']:
            print(f"  {gate['name']}: {gate['center']['lat']:.7f}, {gate['center']['lon']:.7f}")

        if gate_type == 'speed' and 'exit_gates' in course_config:
            print(f"\nExit Gates:")
            for gate in course_config['exit_gates']:
                print(f"  {gate['name']}: {gate['center']['lat']:.7f}, {gate['center']['lon']:.7f}")

        if 'zones' in course_config:
            print(f"\nZone Accuracy Zones: {len(course_config['zones'])} zones")

        return True

    except Exception as e:
        print(f"✗ Error parsing gate file: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests on both gate files"""
    # Test files
    standard_gate_file = "/home/smiley/Downloads/13-43-22-AZgates.CSV"
    speed_gate_file = "/home/smiley/Downloads/00-20-38-AZSpeedgates.CSV"

    results = []

    # Test standard gates
    if os.path.exists(standard_gate_file):
        results.append(test_gate_file(standard_gate_file, 'standard'))
    else:
        print(f"Standard gate file not found: {standard_gate_file}")

    # Test speed gates
    if os.path.exists(speed_gate_file):
        results.append(test_gate_file(speed_gate_file, 'speed'))
    else:
        print(f"Speed gate file not found: {speed_gate_file}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Test Summary: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
