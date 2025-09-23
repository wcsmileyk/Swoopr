#!/usr/bin/env python3
"""
Management command to monitor and analyze error logs
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import os
import re
from datetime import datetime, timedelta
from collections import Counter, defaultdict


class Command(BaseCommand):
    help = 'Monitor and analyze error logs for debugging 500 errors'

    def add_arguments(self, parser):
        parser.add_argument(
            '--tail',
            type=int,
            default=50,
            help='Number of recent log entries to show (default: 50)',
        )
        parser.add_argument(
            '--hours',
            type=int,
            default=24,
            help='Only show errors from the last N hours (default: 24)',
        )
        parser.add_argument(
            '--summary',
            action='store_true',
            help='Show error summary and statistics',
        )
        parser.add_argument(
            '--watch',
            action='store_true',
            help='Watch for new errors in real-time',
        )
        parser.add_argument(
            '--filter',
            type=str,
            help='Filter logs by keyword (case-insensitive)',
        )

    def handle(self, *args, **options):
        tail_count = options['tail']
        hours_back = options['hours']
        show_summary = options['summary']
        watch_mode = options['watch']
        filter_keyword = options['filter']

        logs_dir = settings.BASE_DIR / 'logs'
        error_log = logs_dir / 'errors.log'
        server_log = logs_dir / 'server_errors.log'

        if not error_log.exists() and not server_log.exists():
            self.stdout.write(
                self.style.WARNING('No error logs found. Logs will be created when errors occur.')
            )
            return

        if watch_mode:
            self._watch_logs([error_log, server_log], filter_keyword)
        elif show_summary:
            self._show_summary([error_log, server_log], hours_back)
        else:
            self._show_recent_errors([error_log, server_log], tail_count, hours_back, filter_keyword)

    def _watch_logs(self, log_files, filter_keyword):
        """Watch log files for new entries"""
        self.stdout.write(self.style.SUCCESS('Watching for new errors... (Press Ctrl+C to stop)'))

        try:
            import time
            file_positions = {}

            # Initialize file positions
            for log_file in log_files:
                if log_file.exists():
                    with open(log_file, 'r') as f:
                        f.seek(0, 2)  # Go to end of file
                        file_positions[log_file] = f.tell()

            while True:
                for log_file in log_files:
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            f.seek(file_positions.get(log_file, 0))
                            new_lines = f.readlines()
                            file_positions[log_file] = f.tell()

                            for line in new_lines:
                                if not filter_keyword or filter_keyword.lower() in line.lower():
                                    self.stdout.write(f"[{log_file.name}] {line.strip()}")

                time.sleep(1)

        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS('\nStopped watching logs.'))

    def _show_summary(self, log_files, hours_back):
        """Show error summary and statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        error_types = Counter()
        error_paths = Counter()
        error_times = []
        total_errors = 0

        self.stdout.write(f"\n📊 Error Summary (Last {hours_back} hours)")
        self.stdout.write("=" * 50)

        for log_file in log_files:
            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                for line in f:
                    if 'ERROR' in line:
                        # Try to parse timestamp
                        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                        if timestamp_match:
                            try:
                                log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                                if log_time < cutoff_time:
                                    continue
                                error_times.append(log_time)
                            except ValueError:
                                pass

                        # Extract error type
                        error_match = re.search(r'(\w+Error|\w+Exception)', line)
                        if error_match:
                            error_types[error_match.group(1)] += 1

                        # Extract request path
                        path_match = re.search(r'Request.*?"path": "([^"]+)"', line)
                        if path_match:
                            error_paths[path_match.group(1)] += 1

                        total_errors += 1

        self.stdout.write(f"Total errors: {total_errors}")

        if error_types:
            self.stdout.write(f"\n🔍 Most Common Error Types:")
            for error_type, count in error_types.most_common(10):
                self.stdout.write(f"  {error_type}: {count}")

        if error_paths:
            self.stdout.write(f"\n🌐 Most Problematic Paths:")
            for path, count in error_paths.most_common(10):
                self.stdout.write(f"  {path}: {count}")

        if error_times:
            # Group by hour
            hour_counts = defaultdict(int)
            for error_time in error_times:
                hour_counts[error_time.strftime('%Y-%m-%d %H:00')] += 1

            self.stdout.write(f"\n⏰ Errors by Hour:")
            for hour in sorted(hour_counts.keys())[-24:]:  # Last 24 hours
                self.stdout.write(f"  {hour}: {hour_counts[hour]}")

    def _show_recent_errors(self, log_files, tail_count, hours_back, filter_keyword):
        """Show recent error log entries"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        all_lines = []

        for log_file in log_files:
            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                for line in f:
                    # Filter by time if possible
                    timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                            if log_time < cutoff_time:
                                continue
                        except ValueError:
                            pass

                    # Filter by keyword
                    if filter_keyword and filter_keyword.lower() not in line.lower():
                        continue

                    all_lines.append((log_file.name, line.strip()))

        # Show most recent entries
        recent_lines = all_lines[-tail_count:] if len(all_lines) > tail_count else all_lines

        if not recent_lines:
            self.stdout.write(self.style.WARNING('No matching log entries found.'))
            return

        self.stdout.write(f"\n📋 Recent Errors (Last {len(recent_lines)} entries)")
        self.stdout.write("=" * 70)

        for log_name, line in recent_lines:
            # Color code by log file
            if 'server_errors' in log_name:
                prefix = self.style.ERROR(f"[{log_name}]")
            else:
                prefix = self.style.WARNING(f"[{log_name}]")

            self.stdout.write(f"{prefix} {line}")

        self.stdout.write(f"\n💡 Tip: Use --summary for error statistics")
        self.stdout.write(f"💡 Tip: Use --watch to monitor real-time")
        self.stdout.write(f"💡 Tip: Use --filter <keyword> to search logs")