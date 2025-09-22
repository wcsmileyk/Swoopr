#!/usr/bin/env python3
"""
Test script to verify the improved turn detection algorithm
against the rowen_incorrect_flight.csv data.
"""

import pandas as pd
import numpy as np
import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def test_turn_detection():
    # Load the CSV data
    df = pd.read_csv('/home/smiley/Downloads/rowen_incorrect_flight.csv')

    # Add the t_s column (relative time in seconds) and required column aliases
    # FlySight data is at 5Hz (0.2 second intervals)
    df['t_s'] = np.arange(len(df)) * 0.2

    # Map column names to what FlightManager expects
    df['gspeed'] = df['ground_speed']
    df['AGL'] = df['altitude_agl']
    df['velD'] = df['velocity_down']
    df['velN'] = df['velocity_north']
    df['velE'] = df['velocity_east']
    df['hMSL'] = df['altitude_msl']
    df['hAcc'] = df['h_acc']
    df['vAcc'] = df['v_acc']
    df['sAcc'] = df['s_acc']
    df['numSV'] = df['num_sv']

    print(f"Loaded {len(df)} data points from flight data")
    print(f"Time range: {df['t_s'].iloc[0]:.1f}s to {df['t_s'].iloc[-1]:.1f}s")

    # Use FlightManager's landing detection (which works backwards from end)
    manager = FlightManager()
    landing_idx = manager.get_landing(df)
    print(f"Landing detected by FlightManager at index: {landing_idx} (AGL: {df.iloc[landing_idx]['altitude_agl']:.1f}m)")

    # Also check what would happen if we used the end of the file as landing
    end_idx = len(df) - 1
    print(f"End of file index: {end_idx} (AGL: {df.iloc[end_idx]['altitude_agl']:.1f}m)")

    # For testing the turn detection at 7239, let's use a landing point that makes sense
    # Look for landing after the turn (index 7239 + some time for the swoop to complete)
    test_landing_candidates = df[df.index > 7300]  # After the turn
    if not test_landing_candidates.empty:
        test_landing_idx = test_landing_candidates['altitude_agl'].idxmin()
        print(f"Test landing after turn: index {test_landing_idx} (AGL: {df.iloc[test_landing_idx]['altitude_agl']:.1f}m)")
    else:
        test_landing_idx = landing_idx

    # Show altitude distribution around landing
    print(f"Altitude range: {df['altitude_agl'].min():.1f}m to {df['altitude_agl'].max():.1f}m")

    # Look at data around the expected turn area (7239)
    print(f"\nData around expected turn start (index 7239):")
    print(f"Index 7239: AGL={df.iloc[7239]['altitude_agl']:.1f}m, heading={df.iloc[7239]['heading']:.1f}°")

    # Test the turn detection using the appropriate landing
    print(f"\n" + "="*50)
    print("TESTING WITH CORRECT LANDING")
    print("="*50)

    # Use the test landing that occurs after the turn
    landing_for_test = test_landing_idx
    landing_time = df.iloc[landing_for_test]['t_s']
    search_start_time = landing_time - 60.0
    search_mask = (df['t_s'] >= search_start_time) & (df['t_s'] <= landing_time)
    search_indices = df[search_mask].index.tolist()

    print(f"Using landing at index {landing_for_test}, time {landing_time:.1f}s")
    print(f"Search window: {search_start_time:.1f}s to {landing_time:.1f}s")
    if search_indices:
        print(f"Search indices: {search_indices[0]} to {search_indices[-1]}")
        print(f"Index 7239 in search window: {'YES' if 7239 in search_indices else 'NO'}")
    print(f"Time at index 7239: {df.iloc[7239]['t_s']:.1f}s")
    print(f"Time difference: index 7239 is {df.iloc[7239]['t_s'] - landing_time:.1f}s before/after landing")

    # Let's find the actual lowest altitude points to understand the flight profile
    print(f"\nAltitude analysis:")
    min_alt_indices = df.nsmallest(10, 'altitude_agl').index.tolist()
    print("10 lowest altitude points:")
    for i, idx in enumerate(min_alt_indices):
        print(f"{i+1}. Index {idx}: AGL={df.iloc[idx]['altitude_agl']:.1f}m, time={df.iloc[idx]['t_s']:.1f}s")

    # Look for altitude around index 7239 to confirm this is airborne
    print(f"\nAltitude at index 7239: {df.iloc[7239]['altitude_agl']:.1f}m AGL - {'AIRBORNE' if df.iloc[7239]['altitude_agl'] > 50 else 'NEAR GROUND'}")

    try:
        # Test the fallback turn detection method with the correct landing
        turn_start_idx = manager.find_turn_start_fallback(df, landing_for_test)

        print(f"\nTurn detection results:")
        print(f"Turn start detected at index: {turn_start_idx}")
        print(f"Expected turn start (user identified): 7239")
        print(f"Difference: {turn_start_idx - 7239} records")
        print(f"Detected index in search window: {'YES' if turn_start_idx >= search_indices[0] and turn_start_idx <= search_indices[-1] else 'NO'}")

        # Show headings around the detected turn start
        start_idx = max(0, turn_start_idx - 5)
        end_idx = min(len(df), turn_start_idx + 10)

        print(f"\nHeadings around detected turn start (index {turn_start_idx}):")
        for i in range(start_idx, end_idx):
            marker = " ←" if i == turn_start_idx else "  "
            expected_marker = " (USER)" if i == 7239 else ""
            print(f"Index {i:4d}: heading {df.iloc[i]['heading']:6.1f}°{marker}{expected_marker}")

        # Show headings around the user-identified turn start (7239)
        print(f"\nHeadings around user-identified turn start (index 7239):")
        for i in range(7230, 7250):
            marker = " ←" if i == 7239 else "  "
            print(f"Index {i:4d}: heading {df.iloc[i]['heading']:6.1f}°{marker}")

        # Test the algorithm logic around index 7239
        window_size = 25  # 5 seconds at 5Hz

        # Calculate forward-looking turn rate from index 7239 (as the algorithm does)
        test_idx = 7239
        if test_idx + window_size * 2 < len(df):
            # First window: current to +5 seconds
            heading_start = df.iloc[test_idx]['heading']
            heading_mid = df.iloc[test_idx + window_size]['heading']

            # Handle 360° wraparound
            def heading_diff(h1, h2):
                diff = h2 - h1
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                return diff

            heading_change_1 = heading_diff(heading_start, heading_mid)
            turn_rate_1 = abs(heading_change_1) / (window_size * 0.2)

            # Second window: +5 to +10 seconds
            heading_end = df.iloc[test_idx + window_size * 2]['heading']
            heading_change_2 = heading_diff(heading_mid, heading_end)
            turn_rate_2 = abs(heading_change_2) / (window_size * 0.2)

            # Same direction check
            same_direction = (heading_change_1 * heading_change_2) > 0
            sustained_turn = turn_rate_2 > 4.0  # Half of 8.0 threshold

            print(f"\nTesting algorithm logic at index 7239:")
            print(f"Heading at 7239: {heading_start:.1f}°")
            print(f"Heading at 7239+25: {heading_mid:.1f}°")
            print(f"Heading at 7239+50: {heading_end:.1f}°")
            print(f"First 5s change: {heading_change_1:.1f}° (rate: {turn_rate_1:.1f}°/s)")
            print(f"Second 5s change: {heading_change_2:.1f}° (rate: {turn_rate_2:.1f}°/s)")
            print(f"Same direction: {same_direction}")
            print(f"Sustained turn (>4.0°/s): {sustained_turn}")
            print(f"First window meets threshold (>8.0°/s): {'YES' if turn_rate_1 > 8.0 else 'NO'}")

            # Test the improved logic
            large_continuous_turn = (turn_rate_1 > 15.0 and turn_rate_2 > 10.0)

            print(f"Large continuous turn check (>15°/s & >10°/s): {large_continuous_turn}")

            if turn_rate_1 > 8.0 and ((same_direction and sustained_turn) or large_continuous_turn):
                print(f"✅ This point SHOULD be detected as turn start")
            else:
                print(f"❌ This point would NOT be detected as turn start")

    except Exception as e:
        print(f"Turn detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_turn_detection()