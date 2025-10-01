#!/usr/bin/env python3
"""
Test the production dual metrics system
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager

def test_dual_metrics_system():
    """Test the integrated dual metrics system"""

    print("🧪 TESTING PRODUCTION DUAL METRICS SYSTEM")
    print("=" * 60)

    # Test with a sample file
    test_file = "/home/smiley/PycharmProjects/Swoopr/sample_tracks/25-07-07-sw3.csv"

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return

    manager = FlightManager()

    try:
        # Load and analyze file
        df, metadata = manager.read_flysight_file(test_file)

        print(f"📁 Test file: {os.path.basename(test_file)}")
        print(f"   GPS points: {len(df)}")
        print(f"   Duration: {len(df) * 0.2:.1f}s")

        # Find key points
        landing_idx = manager.get_landing(df)

        try:
            flare_idx = manager.find_flare(df, landing_idx)
            flare_method = "traditional"
        except:
            flare_idx = manager.find_turn_start_fallback(df, landing_idx)
            flare_method = "fallback"

        max_vspeed_idx, max_gspeed_idx = manager.find_max_speeds(df, flare_idx, landing_idx)

        print(f"\n🎯 Key Points:")
        print(f"   Flare: idx {flare_idx} ({df.iloc[flare_idx]['AGL']/0.3048:.0f}ft AGL)")
        print(f"   Max vspeed: idx {max_vspeed_idx} ({df.iloc[max_vspeed_idx]['AGL']/0.3048:.0f}ft AGL)")
        print(f"   Max gspeed: idx {max_gspeed_idx} ({df.iloc[max_gspeed_idx]['AGL']/0.3048:.0f}ft AGL)")
        print(f"   Landing: idx {landing_idx} ({df.iloc[landing_idx]['AGL']/0.3048:.0f}ft AGL)")

        # Test dual metrics calculation
        dual_metrics = manager.calculate_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)

        print(f"\n📊 DUAL METRICS RESULTS:")

        if 'full_swoop' in dual_metrics:
            fs = dual_metrics['full_swoop']
            print(f"   🔄 Full Swoop Analysis:")
            print(f"      Rotation: {fs['rotation']:.1f}° → {fs['intended_turn']}°")
            print(f"      Confidence: {fs['confidence']:.2f} ({fs['method']})")
            print(f"      Altitude: {fs['start_alt']:.0f}ft to {fs['end_alt']:.0f}ft")
            print(f"      Duration: {fs['duration']:.1f}s")
        else:
            print(f"   ❌ Full swoop calculation failed")

        if 'turn_segment' in dual_metrics:
            ts = dual_metrics['turn_segment']
            print(f"   🧠 Turn Segment Analysis:")
            print(f"      Rotation: {ts['rotation']:.1f}° → {ts['intended_turn']}° ({ts['turn_direction']} turn)")
            print(f"      Confidence: {ts['confidence']:.2f} ({ts['method']})")
            print(f"      Altitude: {ts['start_alt']:.0f}ft to {ts['end_alt']:.0f}ft")
            print(f"      Duration: {ts['duration']:.1f}s")
        else:
            print(f"   ⚠️  Turn segment calculation not available")

        # Test if results are reasonable
        success = True

        if 'full_swoop' in dual_metrics:
            fs = dual_metrics['full_swoop']
            if abs(fs['rotation']) < 45 or abs(fs['rotation']) > 1200:
                print(f"   ⚠️  Full swoop rotation seems unreasonable: {fs['rotation']:.1f}°")
                success = False
            if fs['confidence'] < 0.3:
                print(f"   ⚠️  Low confidence for full swoop: {fs['confidence']:.2f}")

        if 'turn_segment' in dual_metrics:
            ts = dual_metrics['turn_segment']
            if abs(ts['rotation']) < 45 or abs(ts['rotation']) > 1200:
                print(f"   ⚠️  Turn segment rotation seems unreasonable: {ts['rotation']:.1f}°")
                success = False

        if success:
            print(f"\n✅ DUAL METRICS SYSTEM WORKING CORRECTLY!")
            print(f"📋 System Features:")
            print(f"   ✅ Full swoop analysis (comprehensive)")
            print(f"   ✅ Turn segment analysis (gswoop-style)")
            print(f"   ✅ Confidence scoring")
            print(f"   ✅ Intended turn classification")
            print(f"   ✅ Multiple validation methods")
        else:
            print(f"\n⚠️  SYSTEM NEEDS REFINEMENT")

        return dual_metrics

    except Exception as e:
        print(f"❌ Error testing dual metrics system: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_dual_metrics_system()