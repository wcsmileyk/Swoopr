#!/usr/bin/env python3
"""
Comprehensive performance profiler for Swoopr flight analysis
Measures CPU, memory, and IO statistics for all processing steps
"""

import os
import sys
import django
import tracemalloc
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from io import StringIO

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Swoopr.settings')
django.setup()

from flights.flight_manager import FlightManager
from flights.models import Flight
from django.contrib.auth.models import User

class PerformanceProfiler:
    def __init__(self):
        self.stats = {}
        self.process = psutil.Process()
        self.start_memory = None
        self.start_io = None

    def get_io_stats(self):
        """Get disk IO statistics"""
        try:
            io_counters = self.process.io_counters()
            return {
                'read_bytes': io_counters.read_bytes,
                'write_bytes': io_counters.write_bytes,
                'read_count': io_counters.read_count,
                'write_count': io_counters.write_count,
            }
        except:
            return None

    def get_memory_info(self):
        """Get memory information"""
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss / 1024 / 1024,  # MB
            'vms': mem_info.vms / 1024 / 1024,  # MB
        }

    @contextmanager
    def profile_operation(self, operation_name, verbose=True):
        """Context manager to profile a specific operation"""
        print(f"\n▶ {operation_name}...")

        # Initial state
        start_time = time.time()
        start_memory = self.get_memory_info()
        start_io = self.get_io_stats()
        tracemalloc.start()

        try:
            yield
        finally:
            # Final state
            elapsed = time.time() - start_time
            end_memory = self.get_memory_info()
            end_io = self.get_io_stats()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Calculate deltas
            memory_delta = {
                'rss': end_memory['rss'] - start_memory['rss'],
                'vms': end_memory['vms'] - start_memory['vms'],
            }

            io_delta = None
            if start_io and end_io:
                io_delta = {
                    'read_bytes': end_io['read_bytes'] - start_io['read_bytes'],
                    'write_bytes': end_io['write_bytes'] - start_io['write_bytes'],
                    'read_count': end_io['read_count'] - start_io['read_count'],
                    'write_count': end_io['write_count'] - start_io['write_count'],
                }

            stat = {
                'elapsed_seconds': elapsed,
                'start_memory_mb': start_memory['rss'],
                'end_memory_mb': end_memory['rss'],
                'peak_memory_mb': peak / 1024 / 1024,
                'memory_delta_mb': memory_delta['rss'],
                'io_read_mb': io_delta['read_bytes'] / 1024 / 1024 if io_delta else None,
                'io_write_mb': io_delta['write_bytes'] / 1024 / 1024 if io_delta else None,
                'io_read_ops': io_delta['read_count'] if io_delta else None,
                'io_write_ops': io_delta['write_count'] if io_delta else None,
            }

            self.stats[operation_name] = stat

            if verbose:
                print(f"  ⏱  Time: {elapsed:.2f}s")
                print(f"  💾 Memory: {start_memory['rss']:.1f}MB → {end_memory['rss']:.1f}MB (Δ {memory_delta['rss']:+.1f}MB, peak {peak/1024/1024:.1f}MB)")
                if io_delta:
                    print(f"  📁 IO: {io_delta['read_bytes']/1024/1024:.1f}MB read, {io_delta['write_bytes']/1024/1024:.1f}MB written")
                    print(f"       {io_delta['read_count']} read ops, {io_delta['write_count']} write ops")

    def generate_report(self):
        """Generate a comprehensive performance report"""
        report = StringIO()
        report.write("\n" + "="*80 + "\n")
        report.write("SWOOPR PERFORMANCE ANALYSIS REPORT\n")
        report.write("="*80 + "\n\n")

        report.write("OPERATION TIMING & RESOURCE USAGE\n")
        report.write("-"*80 + "\n")
        report.write(f"{'Operation':<35} {'Time (s)':<12} {'Memory (MB)':<15} {'IO Read (MB)':<15}\n")
        report.write("-"*80 + "\n")

        total_time = 0
        total_memory_delta = 0
        total_io_read = 0

        for op_name, stats in sorted(self.stats.items(), key=lambda x: x[1]['elapsed_seconds'], reverse=True):
            total_time += stats['elapsed_seconds']
            total_memory_delta += stats['memory_delta_mb']
            if stats['io_read_mb']:
                total_io_read += stats['io_read_mb']

            io_str = f"{stats['io_read_mb']:.1f}" if stats['io_read_mb'] else "N/A"
            mem_str = f"{stats['memory_delta_mb']:+.1f}"

            report.write(f"{op_name:<35} {stats['elapsed_seconds']:<12.2f} {mem_str:<15} {io_str:<15}\n")

        report.write("-"*80 + "\n")
        report.write(f"{'TOTAL':<35} {total_time:<12.2f} {total_memory_delta:+.1f}\n")
        report.write("\n")

        # Bottleneck analysis
        report.write("BOTTLENECK ANALYSIS\n")
        report.write("-"*80 + "\n")

        # Find slowest operations
        sorted_ops = sorted(self.stats.items(), key=lambda x: x[1]['elapsed_seconds'], reverse=True)
        for i, (op_name, stats) in enumerate(sorted_ops[:5], 1):
            pct = (stats['elapsed_seconds'] / total_time) * 100
            report.write(f"{i}. {op_name}: {stats['elapsed_seconds']:.2f}s ({pct:.1f}% of total)\n")

        # Memory intensive operations
        report.write("\nMEMORY-INTENSIVE OPERATIONS\n")
        report.write("-"*80 + "\n")
        sorted_mem = sorted(self.stats.items(), key=lambda x: x[1]['peak_memory_mb'], reverse=True)
        for i, (op_name, stats) in enumerate(sorted_mem[:5], 1):
            report.write(f"{i}. {op_name}: {stats['peak_memory_mb']:.1f}MB peak\n")

        # IO intensive operations
        report.write("\nIO-INTENSIVE OPERATIONS\n")
        report.write("-"*80 + "\n")
        io_ops = [(n, s) for n, s in self.stats.items() if s['io_read_mb'] is not None]
        sorted_io = sorted(io_ops, key=lambda x: x[1]['io_read_mb'], reverse=True)
        for i, (op_name, stats) in enumerate(sorted_io[:5], 1):
            read_mb = stats['io_read_mb'] if stats['io_read_mb'] else 0
            write_mb = stats['io_write_mb'] if stats['io_write_mb'] else 0
            report.write(f"{i}. {op_name}: {read_mb:.1f}MB read, {write_mb:.1f}MB written\n")

        return report.getvalue()


def find_sample_flight_file():
    """Find a sample flight file for testing"""
    # Try sample_tracks first
    sample_tracks = Path("/home/smiley/PycharmProjects/Swoopr/sample_tracks")
    if sample_tracks.exists():
        # Look for files with $FLYS format (valid FlySight files)
        csv_files = list(sample_tracks.glob("*.csv"))
        # Filter for valid FlySight format files
        for f in csv_files:
            try:
                with open(f, 'r') as file:
                    first_line = file.readline()
                    if first_line.startswith('$FLYS'):
                        return f
            except:
                pass
        # If no valid format found, return first file anyway
        if csv_files:
            return csv_files[0]

    # Fallback: search for CSV files in project
    project_root = Path("/home/smiley/PycharmProjects/Swoopr")
    csv_files = list(project_root.rglob("*.csv"))

    if csv_files:
        print(f"Found {len(csv_files)} CSV files:")
        for f in csv_files[:5]:
            print(f"  - {f}")
        return csv_files[0]
    return None


def profile_flight_analysis():
    """Profile the complete flight analysis pipeline"""

    flight_file = find_sample_flight_file()
    if not flight_file:
        print("❌ No sample flight file found!")
        return

    print(f"✅ Found sample flight: {flight_file}")
    print(f"📊 File size: {flight_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Get or create a test pilot
    try:
        pilot = User.objects.first()
        if not pilot:
            print("⚠️  No users found in database. Creating test user...")
            pilot = User.objects.create_user(username='test_pilot', password='test')
    except Exception as e:
        print(f"⚠️  Could not get pilot: {e}")
        pilot = None

    profiler = PerformanceProfiler()

    print("\n" + "="*80)
    print("STARTING PERFORMANCE PROFILING")
    print("="*80)

    with profiler.profile_operation("Initialize FlightManager"):
        fm = FlightManager()

    with profiler.profile_operation("Read FlySight File"):
        df, metadata = fm.read_flysight_file(str(flight_file))

    print(f"\n📈 DataFrame shape: {df.shape} (rows={len(df)}, cols={len(df.columns)})")
    print(f"   Time range: {df['t_s'].min():.1f}s to {df['t_s'].max():.1f}s ({df['t_s'].max() - df['t_s'].min():.1f}s duration)")

    with profiler.profile_operation("Analyze Swoop (No DB)"):
        # Profile analysis without database operations
        try:
            landing_idx = fm.get_landing(df)
            print(f"   Landing detected at index {landing_idx}")

            with profiler.profile_operation("  └─ Landing detection"):
                pass  # Already profiled above

            with profiler.profile_operation("  └─ Flare detection"):
                flare_idx = fm.find_flare(df, landing_idx)

            with profiler.profile_operation("  └─ Max speeds"):
                max_vspeed_idx, max_gspeed_idx = fm.find_max_speeds(df, flare_idx, landing_idx)

            with profiler.profile_operation("  └─ Rotation metrics (traditional)"):
                dual_metrics = fm.calculate_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)

            with profiler.profile_operation("  └─ ML feature extraction"):
                features = fm.extract_ml_features(df, flare_idx, max_gspeed_idx)

            with profiler.profile_operation("  └─ ML prediction"):
                ml_rot, ml_conf, ml_method = fm.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

            print(f"   ✅ Analysis successful")

        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate and print report
    report = profiler.generate_report()
    print(report)

    # Save report
    report_path = Path("/home/smiley/PycharmProjects/Swoopr/PERFORMANCE_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n📄 Report saved to: {report_path}")

    return profiler


if __name__ == '__main__':
    profile_flight_analysis()
