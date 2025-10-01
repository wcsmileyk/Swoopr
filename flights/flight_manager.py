#!/usr/bin/env python3
"""
Consolidated Flight Manager
Combines FlySight file ingestion with swoop analysis
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from django.utils import timezone
from django.contrib.gis.geos import Point
from django.db import transaction
from .models import Flight, GPSPoint

# Constants
MPH_PER_MPS = 2.23694
FT_PER_M = 3.28084


def compute_heading(df):
    """Calculate heading from velocity components"""
    velN, velE = df['velN'].values, df['velE'].values
    heading_degrees = (np.degrees(np.arctan2(velE, velN)) + 360) % 360
    return heading_degrees


def unwrap_deg(headings):
    """Unwrap heading degrees to handle 360° transitions"""
    heading_radians = np.radians(headings)
    unwrapped_radians = np.unwrap(heading_radians)
    unwrapped_degrees = np.degrees(unwrapped_radians)
    return unwrapped_degrees


class InvalidTrackFile(ValueError):
    pass


class SwoopConfig:
    """Configuration for swoop analysis"""
    # vAcc-gated AGL
    vacc_k = 2.5
    ground_tail_s = 90.0
    ground_low_frac = 0.08

    # landing detection
    back_gspeed_min = 6.0
    back_vspeed_min = 6.0
    fwd_gspeed_max = 5.0
    fwd_vspeed_max = 1.0
    fwd_agl_max_m = 10.0
    sustain_stop_s = 0.8
    back_look_s = 120.0
    fwd_confirm_s = 15.0
    min_agl_for_moving_m = 2.0

    # flare / max‑horizontal
    flare_win_vspeed_relax_s = 0.4
    flare_back_win = 30.0
    max_vspeed_threshold = 5.0

    # Windowing relative to flare
    rotation_search_forward_s: float = 5.0

    # Slow-start onset gates
    turn_rate_seed_low_degps: float = 10.0
    min_seed_s: float = 0.6

    # Optional angle-accumulation onset (robust gate near flare)
    angle_gate_deg: float = 20.0
    angle_horizon_s: float = 3.0
    sign_consistency_min: float = 0.7

    # Ongoing turn confirmation and termination
    turn_rate_start_degps: float = 45.0
    turn_rate_stop_degps: float = 10.0
    min_turn_duration_s: float = 0.6

    # Per-step noise floor (degrees per *sample*)
    rotation_eps_step_deg: float = 0.1


class FlightManager:
    """Consolidated manager for FlySight file processing and swoop analysis"""

    def __init__(self, cfg=SwoopConfig):
        self.cfg = cfg

        # ML model components
        self.ml_model = None
        self.ml_feature_names = None
        self.ml_model_loaded = False
        self.load_ml_model()

    def process_file(self, filepath, pilot=None, canopy=None):
        """
        Complete processing pipeline:
        1. Read and parse FlySight CSV
        2. Create Django Flight and GPSPoint objects
        3. Perform swoop analysis
        4. Update Flight with analysis results

        Args:
            filepath: Path to FlySight CSV file
            pilot: User object (required for new user system)
            canopy: Canopy object (optional, will use pilot's primary if not specified)
        """
        if pilot is None:
            raise ValueError("pilot parameter is required")

        try:
            with transaction.atomic():
                # Step 1: Read and parse file
                df, metadata = self.read_flysight_file(filepath)

                # Step 2: Create or get Flight object
                flight = self.create_or_update_flight(filepath, metadata, pilot, canopy)

                # Step 3: Create GPS points
                self.create_gps_points(flight, df)

                # Step 4: Perform swoop analysis
                self.analyze_swoop(flight, df)

                return flight

        except Exception as e:
            # Create flight record with error info
            try:
                flight = self.create_or_update_flight(filepath, {}, pilot, canopy)
                flight.analysis_successful = False
                flight.analysis_error = str(e)
                flight.analyzed_at = timezone.now()
                flight.save()
                return flight
            except Exception:
                raise e

    def read_flysight_file(self, filepath):
        """Read and parse FlySight CSV file - supports multiple formats"""

        # Try to detect file format
        with open(filepath, "r") as f:
            first_line = f.readline().strip()

        # Check if it's a standard CSV with headers
        if not first_line.startswith('$') and 'time' in first_line.lower():
            return self._read_standard_csv(filepath)
        else:
            return self._read_flysight_format(filepath)

    def _read_standard_csv(self, filepath):
        """Read standard CSV format with headers (like gps_00545.csv)"""
        metadata = {}

        # Read the CSV with pandas to auto-detect headers
        df = pd.read_csv(filepath)

        # Skip units row if it exists (second row with units like "(deg)", "(m)", etc.)
        if len(df) > 0 and any('(' in str(val) and ')' in str(val) for val in df.iloc[0].values if pd.notna(val)):
            df = df.iloc[1:].reset_index(drop=True)

        # Standardize column names to match our expected format
        column_mapping = {
            'time': 'time',
            'lat': 'lat',
            'lon': 'lon',
            'hMSL': 'hMSL',
            'velN': 'velN',
            'velE': 'velE',
            'velD': 'velD',
            'hAcc': 'hAcc',
            'vAcc': 'vAcc',
            'sAcc': 'sAcc',
            'numSV': 'numSV',
            'heading': 'heading'
        }

        # Rename columns if they exist
        df.rename(columns=column_mapping, inplace=True)

        # Add missing columns with default values if needed
        required_cols = ['time', 'lat', 'lon', 'hMSL', 'velN', 'velE', 'velD', 'hAcc', 'vAcc', 'sAcc', 'numSV']
        for col in required_cols:
            if col not in df.columns:
                if col == 'numSV':
                    df[col] = 6  # Default satellite count
                elif col in ['hAcc', 'vAcc', 'sAcc']:
                    df[col] = 1.0  # Default accuracy
                else:
                    raise InvalidTrackFile(f"Required column '{col}' not found in CSV")

        # Add $GNSS column if missing (for compatibility)
        if '$GNSS' not in df.columns:
            df['$GNSS'] = '$GPRMC'

        # Convert numeric columns to proper data types
        numeric_cols = ['lat', 'lon', 'hMSL', 'velN', 'velE', 'velD', 'hAcc', 'vAcc', 'sAcc', 'numSV', 'heading']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Parse timestamps - handle both ISO format and epoch
        if df['time'].dtype == 'object':
            # Try ISO format first
            df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True, errors='coerce')
            # If that fails, try without microseconds
            if df['time'].isna().all():
                df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

        # Convert to relative time in seconds
        if df['time'].notna().any():
            start_time = df['time'].iloc[0]
            df['t_s'] = (df['time'] - start_time).dt.total_seconds()
        else:
            # Assume 5Hz data if no valid timestamps
            df['t_s'] = np.arange(len(df)) * 0.2

        # Calculate derived fields
        df['gspeed'] = np.sqrt(df['velN']**2 + df['velE']**2)
        if 'heading' not in df.columns:
            df['heading'] = compute_heading(df)
        df['AGL'] = self.calculate_agl(df)

        # Set metadata
        metadata['format'] = 'standard_csv'
        metadata['device'] = 'unknown'

        return df, metadata

    def _read_flysight_format(self, filepath):
        """Read original FlySight format"""
        # Standard FlySight columns
        cols = ["$GNSS", "time", "lat", "lon", "hMSL", "velN", "velE", "velD",
                "hAcc", "vAcc", "sAcc", "numSV"]

        metadata = {}

        # Read metadata from header
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f):
                if line.startswith('$VAR'):
                    # Parse variable definitions
                    continue
                elif line.startswith('$COL'):
                    # Parse column definitions
                    continue
                elif line.startswith('$UNIT'):
                    # Parse unit definitions
                    continue
                elif line.startswith('$GNSS'):
                    start_row = line_num
                    break
                else:
                    # Extract metadata
                    if ',' in line:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            key = parts[0].replace('$', '')
                            value = ','.join(parts[1:])
                            metadata[key] = value

        if 'start_row' not in locals():
            raise InvalidTrackFile("No $GNSS data found in file")

        # Read GPS data
        df = pd.read_csv(filepath, skiprows=start_row, names=cols)

        # Parse timestamps
        df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%dT%H:%M:%S.%fZ", utc=True, errors='coerce')
        df['t_s'] = df['time'].astype(np.int64) / 1e9

        # Calculate derived fields
        df['gspeed'] = np.sqrt(df['velN']**2 + df['velE']**2)
        df['heading'] = compute_heading(df)
        df['AGL'] = self.calculate_agl(df)

        metadata['format'] = 'flysight'

        return df, metadata

    def calculate_agl(self, df):
        """Calculate Above Ground Level altitude"""
        vAcc = df['vAcc'].dropna()
        median_vAcc = vAcc.median()
        abs_dev = (vAcc - median_vAcc).abs()
        mad_vAcc = abs_dev.median()
        thr = median_vAcc + self.cfg.vacc_k * mad_vAcc
        reliable = (df['vAcc'] <= thr)

        t_s = df['t_s'].astype(float)
        t_end = float(t_s.iloc[-1])
        tail = (t_s >= t_end - self.cfg.ground_tail_s)

        tail_ok = df[reliable & tail]
        if len(tail_ok) > 20:
            ground_hMSL = float(tail_ok['hMSL'].quantile(self.cfg.ground_low_frac))
        else:
            ground_hMSL = float(df['hMSL'].tail(200).min())

        agl = df['hMSL'] - ground_hMSL
        return agl

    def create_or_update_flight(self, filepath, metadata, pilot, canopy=None):
        """Create or update Flight object from file metadata"""
        filename = os.path.basename(filepath)

        # Extract session info from metadata or filename
        device_id = metadata.get('DEVICE', filename.split('-')[0] if '-' in filename else 'unknown')
        session_id = metadata.get('SESSION', filename.replace('.csv', '').replace('.CSV', ''))
        firmware_version = metadata.get('FIRMWARE', '')

        # Use pilot's primary canopy if none specified
        if canopy is None:
            try:
                canopy = pilot.canopies.filter(is_primary=True).first()
            except AttributeError:
                canopy = None

        # Check if user has auto-public flights enabled
        auto_public = getattr(pilot.profile, 'auto_public_flights', False)

        flight, created = Flight.objects.get_or_create(
            pilot=pilot,
            device_id=device_id,
            session_id=session_id,
            defaults={
                'canopy': canopy,
                'firmware_version': firmware_version,
                'filename': filename,
                'analysis_version': 'enhanced_v1',
                'is_public': auto_public,
            }
        )

        if not created:
            # Update existing flight
            flight.canopy = canopy or flight.canopy  # Keep existing if none provided
            flight.firmware_version = firmware_version
            flight.filename = filename
            flight.analysis_version = 'enhanced_v1'
            # Only update privacy if flight is currently private and user has auto-public enabled
            if not flight.is_public and auto_public:
                flight.is_public = True
            flight.save()

        return flight

    def create_gps_points(self, flight, df):
        """Store GPS data as compressed JSON (new efficient approach)"""
        gps_data_list = []

        for idx, row in df.iterrows():
            if pd.isna(row['time']) or pd.isna(row['lat']) or pd.isna(row['lon']):
                continue

            # Convert to timestamp (seconds since epoch)
            timestamp = row['time'].timestamp() if hasattr(row['time'], 'timestamp') else float(row['time'])

            # Handle missing accuracy values
            h_acc = float(row['hAcc']) if pd.notna(row['hAcc']) and row['hAcc'] != '' else None
            v_acc = float(row['vAcc']) if pd.notna(row['vAcc']) and row['vAcc'] != '' else None
            s_acc = float(row['sAcc']) if pd.notna(row['sAcc']) and row['sAcc'] != '' else None

            # Create compact GPS point data
            point_data = {
                'timestamp': timestamp,
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'altitude_msl': float(row['hMSL']),
                'altitude_agl': float(row['AGL']),
                'velocity_north': float(row['velN']),
                'velocity_east': float(row['velE']),
                'velocity_down': float(row['velD']),
                'ground_speed': float(row['gspeed']),
                'heading': float(row['heading']),
                'h_acc': h_acc,
                'v_acc': v_acc,
                's_acc': s_acc,
                'num_sv': int(row['numSV'])
            }
            gps_data_list.append(point_data)

        # Store as compressed JSON
        flight.store_gps_data(gps_data_list)

    def _create_gps_points_legacy(self, flight, df):
        """Legacy GPS point creation - will be removed after full migration"""
        # Clear existing GPS points for this flight
        flight.gps_points.all().delete()

        gps_points = []
        for idx, row in df.iterrows():
            if pd.isna(row['time']) or pd.isna(row['lat']) or pd.isna(row['lon']):
                continue

            # Handle missing accuracy values
            h_acc = float(row['hAcc']) if pd.notna(row['hAcc']) and row['hAcc'] != '' else None
            v_acc = float(row['vAcc']) if pd.notna(row['vAcc']) and row['vAcc'] != '' else None
            s_acc = float(row['sAcc']) if pd.notna(row['sAcc']) and row['sAcc'] != '' else None

            point = GPSPoint(
                flight=flight,
                timestamp=row['time'],
                location=Point(float(row['lon']), float(row['lat'])),
                altitude_msl=float(row['hMSL']),
                altitude_agl=float(row['AGL']),
                velocity_north=float(row['velN']),
                velocity_east=float(row['velE']),
                velocity_down=float(row['velD']),
                ground_speed=float(row['gspeed']),
                heading=float(row['heading']),
                horizontal_accuracy=h_acc,
                vertical_accuracy=v_acc,
                speed_accuracy=s_acc,
                num_satellites=int(row['numSV'])
            )
            gps_points.append(point)

        # Bulk create for performance
        GPSPoint.objects.bulk_create(gps_points, batch_size=1000)

    def analyze_swoop(self, flight, df):
        """Perform comprehensive swoop analysis"""
        try:
            # Run swoop detection
            landing_idx = self.get_landing(df)

            # Try traditional flare detection first
            try:
                flare_idx = self.find_flare(df, landing_idx)
                flare_method = "traditional"
            except (ValueError, IndexError):
                # Fallback: no clear flare detected, use turn detection approach
                flare_idx = self.find_turn_start_fallback(df, landing_idx)
                flare_method = "turn_detection"

            max_vspeed_idx, max_gspeed_idx = self.find_max_speeds(df, flare_idx, landing_idx)

            # Calculate dual rotation metrics
            dual_metrics = self.calculate_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)

            # Use full swoop rotation as primary (maintains compatibility)
            if 'full_swoop' in dual_metrics:
                fs = dual_metrics['full_swoop']
                turn_rotation = fs['rotation']
                intended_turn = fs['intended_turn']
                rotation_confidence = fs['confidence']
                rotation_method = fs['method']
            else:
                # Fallback to legacy method
                turn_rotation, intended_turn, rotation_confidence, rotation_method = self.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

            # Get ML prediction
            ml_rotation, ml_intended, ml_confidence, ml_method = self.get_rotation_with_ml_enhancement(df, flare_idx, max_gspeed_idx)

            rollout_start_idx, rollout_end_idx = self.get_roll_out(df, max_vspeed_idx, max_gspeed_idx, landing_idx)

            # Calculate metrics
            flare_time = df.iloc[flare_idx]['t_s']
            max_gspeed_time = df.iloc[max_gspeed_idx]['t_s']
            landing_time = df.iloc[landing_idx]['t_s']
            rollout_time = df.iloc[rollout_end_idx]['t_s'] - df.iloc[rollout_start_idx]['t_s']

            max_vspeed_ms = abs(df.iloc[max_vspeed_idx]['velD'])
            max_gspeed_ms = df.iloc[max_gspeed_idx]['gspeed']

            # Update flight with analysis results
            flight.is_swoop = True
            flight.landing_detected = True
            flight.analysis_successful = True

            # Store indices
            flight.landing_idx = landing_idx
            flight.flare_idx = flare_idx
            flight.max_vspeed_idx = max_vspeed_idx
            flight.max_gspeed_idx = max_gspeed_idx
            flight.rollout_start_idx = rollout_start_idx
            flight.rollout_end_idx = rollout_end_idx

            # Store detection method for reference
            flight.flare_detection_method = flare_method

            # Store calculated metrics (in metric units)
            flight.turn_rotation = turn_rotation
            flight.turn_rotation_confidence = rotation_confidence
            flight.turn_rotation_method = rotation_method
            flight.intended_turn = intended_turn
            flight.turn_direction = "left" if turn_rotation < 0 else "right"

            # Store turn segment metrics (gswoop-style)
            if 'turn_segment' in dual_metrics:
                ts = dual_metrics['turn_segment']
                flight.turn_segment_rotation = ts['rotation']
                flight.turn_segment_confidence = ts['confidence']
                flight.turn_segment_method = ts['method']
                flight.turn_segment_intended = ts['intended_turn']
                flight.turn_segment_start_alt = ts['start_alt']
                flight.turn_segment_end_alt = ts['end_alt']
                flight.turn_segment_duration = ts['duration']
                # gswoop_difference will be populated later when compared with actual gswoop data
                flight.gswoop_difference = None
            else:
                # Clear turn segment fields if not calculated
                flight.turn_segment_rotation = None
                flight.turn_segment_confidence = None
                flight.turn_segment_method = None
                flight.turn_segment_intended = None
                flight.turn_segment_start_alt = None
                flight.turn_segment_end_alt = None
                flight.turn_segment_duration = None
                flight.gswoop_difference = None

            # Store ML prediction metrics
            flight.ml_rotation = ml_rotation
            flight.ml_rotation_confidence = ml_confidence
            flight.ml_rotation_method = ml_method
            flight.ml_intended_turn = ml_intended
            flight.max_vertical_speed_ms = max_vspeed_ms
            flight.max_ground_speed_ms = max_gspeed_ms

            # Store legacy imperial values for migration compatibility
            flight.max_vertical_speed_mph = max_vspeed_ms * MPH_PER_MPS
            flight.max_ground_speed_mph = max_gspeed_ms * MPH_PER_MPS
            flight.turn_time = max_gspeed_time - flare_time
            flight.rollout_time = rollout_time

            # Store altitudes
            flight.exit_altitude_agl = df['AGL'].max()
            flight.flare_altitude_agl = df.iloc[flare_idx]['AGL']
            flight.max_vspeed_altitude_agl = df.iloc[max_vspeed_idx]['AGL']
            flight.max_gspeed_altitude_agl = df.iloc[max_gspeed_idx]['AGL']
            flight.rollout_start_altitude_agl = df.iloc[rollout_start_idx]['AGL'] if rollout_start_idx is not None else None
            flight.rollout_end_altitude_agl = df.iloc[rollout_end_idx]['AGL'] if rollout_end_idx is not None else None
            flight.landing_altitude_agl = df.iloc[landing_idx]['AGL']

            # Calculate average altitude during swoop (flare to landing)
            swoop_altitudes = df.iloc[rollout_end_idx:landing_idx+1]['AGL']
            flight.swoop_avg_altitude_agl = swoop_altitudes.mean()

            # Calculate entry gate speed (speed at flare initiation) - store in metric
            flare_gspeed_ms = df.iloc[flare_idx]['gspeed']
            flight.entry_gate_speed_mps = flare_gspeed_ms
            flight.entry_gate_speed_mph = flare_gspeed_ms * MPH_PER_MPS  # Legacy

            # Store timing
            flight.total_flight_time = landing_time - df.iloc[0]['t_s']
            flight.swoop_start_time = df.iloc[flare_idx]['time']
            flight.swoop_end_time = df.iloc[landing_idx]['time']

            flight.analyzed_at = timezone.now()
            flight.save()

            # Calculate accuracy metrics directly from DataFrame
            if flare_idx is not None and landing_idx is not None:
                swoop_slice = df.iloc[flare_idx:landing_idx + 1]

                # Calculate accuracy metrics
                h_acc_values = swoop_slice['hAcc'].dropna()
                v_acc_values = swoop_slice['vAcc'].dropna()
                s_acc_values = swoop_slice['sAcc'].dropna()

                flight.swoop_avg_horizontal_accuracy = h_acc_values.mean() if len(h_acc_values) > 0 else None
                flight.swoop_avg_vertical_accuracy = v_acc_values.mean() if len(v_acc_values) > 0 else None
                flight.swoop_avg_speed_accuracy = s_acc_values.mean() if len(s_acc_values) > 0 else None

            # Calculate swoop distance directly from DataFrame
            if rollout_end_idx is not None and landing_idx is not None:
                try:
                    rollout_end_row = df.iloc[rollout_end_idx]
                    landing_row = df.iloc[landing_idx]

                    # Calculate distance using Haversine formula
                    import math
                    lat1, lon1 = math.radians(rollout_end_row['lat']), math.radians(rollout_end_row['lon'])
                    lat2, lon2 = math.radians(landing_row['lat']), math.radians(landing_row['lon'])

                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    c = 2 * math.asin(math.sqrt(a))
                    distance_m = 6371000 * c  # Earth radius in meters

                    # Store in metric units
                    flight.swoop_distance_m = distance_m
                    flight.swoop_distance_ft = distance_m * 3.28084  # Legacy
                except (IndexError, TypeError, KeyError):
                    flight.swoop_distance_m = None
                    flight.swoop_distance_ft = None

            flight.save()

        except Exception as e:
            flight.is_swoop = False
            flight.analysis_successful = False
            flight.analysis_error = str(e)
            flight.analyzed_at = timezone.now()
            flight.save()
            raise

    # ============== SWOOP ANALYSIS METHODS ==============
    # These are adapted from the original track_manager.py

    def get_landing(self, df):
        """Detect landing point using original algorithm"""
        d, c = df, self.cfg
        t = d['t_s'].astype(float).to_numpy()
        n = len(d)
        dt = float(np.nanmedian(np.diff(t))) if n > 1 else 0.2
        fs = 1.0/dt if (dt and np.isfinite(dt) and dt > 0) else 5.0
        win_stop = max(1, int(round(c.sustain_stop_s * fs)))

        gspeed = d["gspeed"].to_numpy().astype(float)
        vspeed = d["velD"].to_numpy().astype(float)
        agl = d["AGL"].to_numpy().astype(float)

        # vAcc gate
        vAcc = d['vAcc'].dropna()
        median_vAcc = vAcc.median()
        mad_vAcc = ((vAcc - median_vAcc).abs()).median() or 1.0
        thr = median_vAcc + c.vacc_k * mad_vAcc
        good_acc = (d['vAcc'] <= thr).fillna(True).to_numpy()

        moving = (gspeed > c.back_gspeed_min) & (np.abs(vspeed) > c.back_vspeed_min)
        if c.min_agl_for_moving_m is not None:
            moving &= (agl >= c.min_agl_for_moving_m)

        stopped = (gspeed < c.fwd_gspeed_max) & (np.abs(vspeed) < c.fwd_vspeed_max) & (agl < c.fwd_agl_max_m) & good_acc

        # Stage 1: last moving anchor within back_look_s
        t_end = t[-1]
        recent = (t >= t_end - c.back_look_s)
        idx_recent = np.where(moving & recent)[0]
        if idx_recent.size:
            anchor = int(idx_recent[-1])
        else:
            idx_all = np.where(moving)[0]
            anchor = int(idx_all[-1]) if idx_all.size else n-1

        # Stage 2: sustained stop after anchor
        fwd_t1 = t[anchor] + c.fwd_confirm_s
        fwd_mask = (np.arange(n) >= anchor) & (t <= fwd_t1)
        if win_stop > 1:
            stopped_roll = pd.Series(stopped.astype(int)).rolling(win_stop, min_periods=win_stop).sum().to_numpy()
            idx = np.where((stopped_roll >= win_stop) & fwd_mask)[0]
        else:
            idx = np.where(stopped & fwd_mask)[0]

        if idx.size:
            landing_idx = int(idx[0])
        else:
            # fallback
            if win_stop > 1:
                idx_any = np.where(stopped_roll >= win_stop)[0]
            else:
                idx_any = np.where(stopped)[0]
            idx_any = idx_any[idx_any >= anchor]
            landing_idx = int(idx_any[0]) if idx_any.size else int(np.arange(anchor, n)[np.argmin(gspeed[anchor:]+np.abs(vspeed[anchor:]))])

        return landing_idx

    def find_flare(self, df, landing_idx):
        """Find flare point using original algorithm"""
        cfg = self.cfg
        t_s = df['t_s'].to_numpy().astype(float)
        vspeed = df['velD'].to_numpy().astype(float)

        mask = (t_s >= t_s[landing_idx] - cfg.flare_back_win) & (t_s <= t_s[landing_idx])
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            raise ValueError("No samples in the window")

        sample_interval_s = np.nanmedian(np.diff(t_s[idxs])) if idxs.size > 1 else 0.2
        win_vspeed = max(1, int(round(cfg.flare_win_vspeed_relax_s / sample_interval_s)))

        dvspeed = np.diff(vspeed, prepend=vspeed[0])
        sustained_slow_vspeed = pd.Series((dvspeed >= 0).astype(int)).rolling(win_vspeed, min_periods=win_vspeed).sum().to_numpy() >= win_vspeed
        vspeed_gate = vspeed <= cfg.max_vspeed_threshold

        # Add minimum altitude requirement for valid flare detection
        agl = df['AGL'].to_numpy().astype(float)
        min_flare_altitude = 75.0  # meters AGL
        altitude_gate = agl >= min_flare_altitude

        flare_candidates = np.where(sustained_slow_vspeed & vspeed_gate & altitude_gate & mask)[0]
        if not flare_candidates.any():
            # Fallback: try without vspeed gate but keep altitude requirement
            flare_candidates = np.where(sustained_slow_vspeed & altitude_gate & mask)[0]
        if not flare_candidates.any():
            # Final fallback: original logic without altitude gate
            flare_candidates = np.where(sustained_slow_vspeed & mask)[0]

        flare = int(flare_candidates[0])
        return flare

    def find_turn_start_fallback(self, df, landing_idx):
        """
        Improved fallback method to find turn start for direct-entry swoops.
        Detects sustained heading changes rather than looking for perpendicular approaches.
        """
        t_s = df['t_s'].to_numpy().astype(float)

        # Look back further for direct entry swoops (60 seconds)
        landing_time = t_s[landing_idx]
        search_start_time = landing_time - 60.0

        # Find indices within search window
        search_mask = (t_s >= search_start_time) & (t_s <= landing_time)
        search_idxs = np.where(search_mask)[0]

        if len(search_idxs) < 50:  # Need more data for this analysis
            fallback_idx = max(0, landing_idx - int(60 * 5))  # 60 seconds at 5Hz
            return fallback_idx

        # Calculate headings using stored heading data if available, otherwise compute
        if 'heading' in df.columns:
            all_headings = df['heading'].to_numpy().astype(float)
            headings = all_headings[search_idxs]
        else:
            velN = df['velN'].to_numpy().astype(float)
            velE = df['velE'].to_numpy().astype(float)
            headings = np.degrees(np.arctan2(velE[search_idxs], velN[search_idxs]))

        # Function to calculate heading difference handling 360° wrap
        def heading_diff(h1, h2):
            diff = h2 - h1
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            return diff

        # Look for sustained turn initiation using heading change rate
        window_size = 25  # 5-second analysis windows
        turn_rate_threshold = 8.0  # degrees per second sustained turn

        best_turn_start = None
        max_sustained_rate = 0

        for i in range(0, len(headings) - window_size * 2, 5):  # Check every second
            if i + window_size * 2 >= len(headings):
                break

            # Analyze heading change rate in current window
            window_start = headings[i]
            window_end = headings[i + window_size]
            heading_change = heading_diff(window_start, window_end)
            turn_rate = abs(heading_change) / (window_size * 0.2)  # degrees per second

            # Check if this is followed by sustained turning
            if turn_rate > turn_rate_threshold:
                # Look ahead to confirm sustained turn (next 5 seconds)
                next_window_start = headings[i + window_size]
                next_window_end = headings[i + window_size * 2]
                next_heading_change = heading_diff(next_window_start, next_window_end)
                next_turn_rate = abs(next_heading_change) / (window_size * 0.2)

                # Check if turn continues in same direction or is a sustained large turn
                same_direction = (heading_change * next_heading_change) > 0
                sustained_turn = next_turn_rate > (turn_rate_threshold * 0.5)  # At least half the rate

                # For very high turn rates, relax the same direction requirement
                # This handles continuous turns that cross compass boundaries
                large_continuous_turn = (turn_rate > 15.0 and next_turn_rate > 10.0)

                if (same_direction and sustained_turn) or large_continuous_turn:
                    combined_rate = (turn_rate + next_turn_rate) / 2
                    if combined_rate > max_sustained_rate:
                        max_sustained_rate = combined_rate
                        best_turn_start = search_idxs[i]

        # If we found a good turn start, use it
        if best_turn_start is not None:
            return best_turn_start

        # Fallback: look for any significant heading change
        for i in range(0, len(headings) - 10, 5):
            if i + 10 >= len(headings):
                break

            start_heading = headings[i]
            end_heading = headings[i + 10]
            change = abs(heading_diff(start_heading, end_heading))

            # Look for 20+ degree change over 2 seconds
            if change > 20:
                return search_idxs[i]

        # Final fallback: use start of search window
        return search_idxs[0]

    def find_max_speeds(self, df, flare_idx, landing_idx):
        """Find maximum vertical and ground speed points"""
        t_s = df['t_s'].to_numpy().astype(float)
        vspeed = df['velD'].to_numpy().astype(float)
        gspeed = df['gspeed'].to_numpy().astype(float)
        mask = (t_s >= t_s[flare_idx]) & (t_s <= t_s[landing_idx])
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            raise ValueError("No samples in the window")

        mvspeed_rel = int(np.nanargmax(vspeed[idxs]))
        max_vspeed_idx = int(idxs[mvspeed_rel])

        right_of_max_vspeed = idxs[idxs > max_vspeed_idx]
        if right_of_max_vspeed.size == 0:
            max_gspeed_idx = int(idxs[np.nanargmax(gspeed[idxs])])
            return max_vspeed_idx, max_gspeed_idx
        mgspeed_rel = int(np.nanargmax(gspeed[right_of_max_vspeed]))
        max_gspeed_idx = int(right_of_max_vspeed[mgspeed_rel])
        return max_vspeed_idx, max_gspeed_idx

    def get_rotation(self, df, flare_idx, max_gspeed_idx):
        """Calculate turn rotation with improved algorithm"""
        try:
            # Use improved rotation detection
            rotation, intended_turn, confidence, method = self._improved_rotation_detection(df, flare_idx, max_gspeed_idx)
            return rotation

        except Exception as e:
            # Fallback to original algorithm if improved version fails
            return self._get_rotation_legacy(df, flare_idx, max_gspeed_idx)

    def get_rotation_with_metadata(self, df, flare_idx, max_gspeed_idx):
        """Calculate turn rotation with all metadata"""
        try:
            # Use improved rotation detection
            rotation, intended_turn, confidence, method = self._improved_rotation_detection(df, flare_idx, max_gspeed_idx)
            return rotation, intended_turn, confidence, method

        except Exception as e:
            # Fallback to original algorithm if improved version fails
            rotation = self._get_rotation_legacy(df, flare_idx, max_gspeed_idx)
            return rotation, None, 0.3, "legacy"

    def calculate_dual_rotation_metrics(self, df, flare_idx, max_gspeed_idx, landing_idx):
        """
        Calculate both full swoop and gswoop-style turn segment rotation metrics
        Returns dict with both 'full_swoop' and 'turn_segment' results
        """
        results = {}

        # 1. Full Swoop Rotation (existing comprehensive method)
        try:
            full_rotation, intended_turn, confidence, method = self.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)
            results['full_swoop'] = {
                'rotation': full_rotation,
                'intended_turn': intended_turn,
                'confidence': confidence,
                'method': method,
                'start_alt': df.iloc[flare_idx]['AGL'] / 0.3048,
                'end_alt': df.iloc[max_gspeed_idx]['AGL'] / 0.3048,
                'duration': (max_gspeed_idx - flare_idx) * 0.2,
            }
        except Exception as e:
            print(f"Full swoop rotation calculation failed: {e}")

        # 2. gswoop-style Turn Segment Rotation
        try:
            turn_segment_result = self._calculate_turn_segment_rotation(df, landing_idx)
            if turn_segment_result:
                results['turn_segment'] = turn_segment_result
        except Exception as e:
            print(f"Turn segment rotation calculation failed: {e}")

        return results

    def _calculate_turn_segment_rotation(self, df, landing_idx):
        """
        Calculate gswoop-style turn segment rotation using altitude-based boundary detection
        Attempts to find turn initiation and rollout end points similar to gswoop methodology
        """

        # Estimate turn boundaries using altitude and flight characteristics
        # This is a simplified approach - we can refine with more gswoop comparison data

        # Find potential turn start: look for altitude where turn rate increases significantly
        turn_start_idx = self._estimate_turn_start_altitude(df, landing_idx)

        # Find potential rollout end: look for low altitude where heading stabilizes
        rollout_end_idx = self._estimate_rollout_end_altitude(df, landing_idx)

        if turn_start_idx is None or rollout_end_idx is None or turn_start_idx >= rollout_end_idx:
            return None

        # Calculate rotation over this segment
        turn_segment = df.iloc[turn_start_idx:rollout_end_idx+1]
        headings = turn_segment['heading'].values

        if len(headings) < 2:
            return None

        # Calculate intelligent rotation following the actual flight path
        rotation, turn_direction = self._calculate_intelligent_rotation(headings)

        # Classify intended turn
        abs_rotation = abs(rotation)
        if abs_rotation < 150:
            intended_turn = 90
        elif abs_rotation < 350:
            intended_turn = 270
        elif abs_rotation < 550:
            intended_turn = 450
        else:
            intended_turn = 630

        # Calculate confidence based on turn characteristics
        confidence = self._calculate_turn_segment_confidence(headings, rotation, intended_turn)

        return {
            'rotation': rotation,
            'intended_turn': intended_turn,
            'confidence': confidence,
            'method': 'altitude_based_segment',
            'start_alt': df.iloc[turn_start_idx]['AGL'] / 0.3048,
            'end_alt': df.iloc[rollout_end_idx]['AGL'] / 0.3048,
            'duration': len(headings) * 0.2,
            'turn_direction': turn_direction
        }

    def _estimate_turn_start_altitude(self, df, landing_idx):
        """Estimate turn start point using altitude and turn rate analysis"""
        # Look for point where heading changes start increasing significantly
        # Typically around 400-800ft AGL for swoops

        min_alt_m = 120 * 0.3048  # ~400ft minimum
        max_alt_m = 800 * 0.3048  # ~800ft maximum

        candidates = df[(df['AGL'] >= min_alt_m) & (df['AGL'] <= max_alt_m)].copy()

        if len(candidates) < 10:
            return None

        # Calculate heading change rate for each candidate point
        best_idx = None
        max_turn_rate = 0

        for idx in candidates.index[5:-5]:  # Avoid edges
            window = df.iloc[idx-5:idx+6]  # 11-point window
            headings = window['heading'].values

            # Calculate average turn rate in this window
            turn_rate = self._calculate_avg_turn_rate(headings)

            if turn_rate > max_turn_rate and turn_rate > 2.0:  # At least 2°/point
                max_turn_rate = turn_rate
                best_idx = idx

        return best_idx

    def _estimate_rollout_end_altitude(self, df, landing_idx):
        """Estimate rollout end point - low altitude where heading stabilizes"""
        # Look for point where heading changes decrease significantly
        # Typically around 10-50ft AGL

        min_alt_m = 3 * 0.3048   # ~10ft minimum
        max_alt_m = 50 * 0.3048  # ~50ft maximum

        candidates = df[(df['AGL'] >= min_alt_m) & (df['AGL'] <= max_alt_m)].copy()

        if len(candidates) < 5:
            # If no candidates in range, use point close to landing
            return max(0, landing_idx - 10)

        # Find point with lowest turn rate
        best_idx = None
        min_turn_rate = float('inf')

        for idx in candidates.index[2:-2]:  # Avoid edges
            window = df.iloc[idx-2:idx+3]  # 5-point window
            headings = window['heading'].values

            turn_rate = self._calculate_avg_turn_rate(headings)

            if turn_rate < min_turn_rate:
                min_turn_rate = turn_rate
                best_idx = idx

        return best_idx

    def _calculate_avg_turn_rate(self, headings):
        """Calculate average absolute turn rate for a heading sequence"""
        if len(headings) < 2:
            return 0

        total_change = 0
        valid_changes = 0

        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]

            # Normalize to [-180, 180]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            # Filter out GPS noise
            if abs(diff) <= 90:
                total_change += abs(diff)
                valid_changes += 1

        return total_change / max(1, valid_changes)

    def _calculate_intelligent_rotation(self, headings):
        """Calculate rotation by following the actual flight path"""
        if len(headings) < 2:
            return 0, "unknown"

        total_left_turn = 0
        total_right_turn = 0

        for i in range(1, len(headings)):
            prev_heading = headings[i-1]
            curr_heading = headings[i]

            diff = curr_heading - prev_heading

            # Normalize to [-180, 180]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            # Filter out GPS noise
            if abs(diff) <= 90:
                if diff < 0:
                    total_left_turn += abs(diff)
                else:
                    total_right_turn += abs(diff)

        # Determine dominant direction and calculate final rotation
        if total_left_turn > total_right_turn:
            net_rotation = -(total_left_turn - total_right_turn)
            direction = "left"
        else:
            net_rotation = total_right_turn - total_left_turn
            direction = "right"

        # For multi-rotation turns, add full rotations
        total_rotation = total_left_turn + total_right_turn
        if total_rotation > 270:
            estimated_full_rotations = int(total_rotation / 360)
            if direction == "left":
                final_rotation = -(abs(net_rotation) + estimated_full_rotations * 360)
            else:
                final_rotation = abs(net_rotation) + estimated_full_rotations * 360
        else:
            final_rotation = net_rotation

        return final_rotation, direction

    def _calculate_turn_segment_confidence(self, headings, rotation, intended_turn):
        """Calculate confidence score for turn segment analysis"""
        confidence = 0.5  # Base confidence

        # Higher confidence for smooth heading progression
        heading_smoothness = self._calculate_heading_smoothness(headings)
        confidence += heading_smoothness * 0.3

        # Higher confidence if rotation is close to standard turn
        distance_to_standard = abs(abs(rotation) - intended_turn)
        if distance_to_standard < 30:
            confidence += 0.2
        elif distance_to_standard < 60:
            confidence += 0.1

        # Higher confidence for reasonable turn duration
        duration = len(headings) * 0.2
        if 3 <= duration <= 15:
            confidence += 0.2

        return max(0.1, min(1.0, confidence))

    def _calculate_heading_smoothness(self, headings):
        """Calculate smoothness of heading progression (0-1 scale)"""
        if len(headings) < 3:
            return 0.5

        # Count direction changes and large jumps
        direction_changes = 0
        large_jumps = 0
        last_diff = None

        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]

            # Normalize
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360

            if abs(diff) > 45:  # Large jump
                large_jumps += 1

            if last_diff is not None:
                if (last_diff > 0) != (diff > 0):  # Direction change
                    direction_changes += 1

            last_diff = diff

        # Calculate smoothness score
        max_changes = len(headings) - 2
        if max_changes <= 0:
            return 0.5

        smoothness = 1.0 - (direction_changes + large_jumps) / max_changes
        return max(0.0, min(1.0, smoothness))

    def load_ml_model(self):
        """Load the trained ML model for rotation prediction"""
        try:
            model_path = Path(__file__).parent / 'rotation_prediction_model.pkl'
            if model_path.exists():
                model_data = joblib.load(model_path)
                self.ml_model = model_data['model']
                self.ml_feature_names = model_data['feature_names']
                self.ml_model_loaded = True
                print(f"✅ ML rotation model loaded (improvement: {model_data.get('improvement', 0):+.1f}%)")
            else:
                print(f"⚠️  ML model not found: {model_path}")
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")

    def extract_ml_features(self, df, flare_idx, max_gspeed_idx):
        """Extract features for ML prediction (matching training format)"""
        features = {}

        # Basic flight characteristics
        features['flight_duration'] = len(df) * 0.2
        features['turn_duration'] = (max_gspeed_idx - flare_idx) * 0.2

        # Altitude features
        features['entry_altitude'] = df.iloc[flare_idx]['AGL'] / 0.3048  # Convert to feet
        features['max_gspeed_altitude'] = df.iloc[max_gspeed_idx]['AGL'] / 0.3048
        features['altitude_loss'] = (df.iloc[flare_idx]['AGL'] - df.iloc[max_gspeed_idx]['AGL']) / 0.3048

        # Speed features
        features['entry_speed'] = df.iloc[flare_idx]['gspeed'] * 2.23694  # Convert to mph
        features['max_vspeed'] = abs(df.iloc[max_gspeed_idx]['velD']) * 2.23694
        features['max_gspeed'] = df.iloc[max_gspeed_idx]['gspeed'] * 2.23694

        # Heading analysis
        turn_data = df[flare_idx:max_gspeed_idx+1]
        headings = turn_data['heading'].values

        if len(headings) >= 2:
            features['heading_start'] = headings[0]
            features['heading_end'] = headings[-1]

            # Calculate net heading change
            net_change = headings[-1] - headings[0]
            while net_change > 180:
                net_change -= 360
            while net_change < -180:
                net_change += 360
            features['net_heading_change'] = net_change
        else:
            features['heading_start'] = 0
            features['heading_end'] = 0
            features['net_heading_change'] = 0

        return features

    def predict_ml_rotation(self, df, flare_idx, max_gspeed_idx):
        """Predict rotation using ML model"""
        if not self.ml_model_loaded:
            return None, 0.0, "ml_unavailable"

        try:
            # Extract features
            features = self.extract_ml_features(df, flare_idx, max_gspeed_idx)

            # Create feature vector in correct order
            feature_vector = np.array([[features[name] for name in self.ml_feature_names]])

            # Predict
            ml_rotation = self.ml_model.predict(feature_vector)[0]

            # Calculate confidence based on reasonable rotation range
            confidence = min(1.0, max(0.3, 1.0 - abs(ml_rotation) / 1200))

            return ml_rotation, confidence, "ml_enhanced"

        except Exception as e:
            print(f"ML prediction error: {e}")
            return None, 0.0, "ml_error"

    def get_rotation_with_ml_enhancement(self, df, flare_idx, max_gspeed_idx):
        """Get rotation with ML enhancement as primary method"""

        # Try ML prediction first
        ml_rotation, ml_confidence, ml_method = self.predict_ml_rotation(df, flare_idx, max_gspeed_idx)

        if ml_rotation is not None and ml_confidence > 0.4:
            # Use ML prediction if confident enough

            # Classify ML prediction to intended turn
            abs_rotation = abs(ml_rotation)
            if abs_rotation < 150:
                ml_intended = 90
            elif abs_rotation < 350:
                ml_intended = 270
            elif abs_rotation < 550:
                ml_intended = 450
            elif abs_rotation < 750:
                ml_intended = 630
            elif abs_rotation < 950:
                ml_intended = 810
            else:
                ml_intended = 990

            return ml_rotation, ml_intended, ml_confidence, ml_method
        else:
            # Fallback to traditional algorithm
            return self.get_rotation_with_metadata(df, flare_idx, max_gspeed_idx)

    def _improved_rotation_detection(self, df, flare_idx, max_gspeed_idx):
        """Enhanced rotation detection with multiple validation methods"""

        # Extract turn data
        turn_data = df[flare_idx:max_gspeed_idx+1].copy()
        headings = turn_data['heading'].values

        if len(headings) < 3:
            return 270.0, 270, 0.5, "default"  # Default fallback

        # Method 1: Smoothed heading approach to reduce GPS noise
        def smooth_headings(headings, window_size=5):
            """Smooth headings using a moving average, handling 360° wraparound"""
            # Convert to complex representation to handle wraparound
            complex_headings = np.exp(1j * np.deg2rad(headings))

            # Smooth in complex domain
            if len(complex_headings) >= window_size:
                # Use a simple moving average
                smoothed_complex = np.convolve(complex_headings,
                                             np.ones(window_size)/window_size,
                                             mode='same')
            else:
                smoothed_complex = complex_headings

            # Convert back to angles
            smoothed_headings = np.rad2deg(np.angle(smoothed_complex))
            smoothed_headings = (smoothed_headings + 360) % 360

            return smoothed_headings

        smooth_hdg = smooth_headings(headings)

        # Method 2: Progressive heading tracking with outlier rejection
        def progressive_heading_analysis(headings):
            """Track heading changes progressively, rejecting outliers"""

            if len(headings) < 2:
                return 0, 0

            total_rotation = 0
            prev_heading = headings[0]

            for heading in headings[1:]:
                # Calculate angular difference
                diff = heading - prev_heading

                # Normalize to [-180, 180]
                while diff > 180:
                    diff -= 360
                while diff < -180:
                    diff += 360

                # Outlier rejection: reject changes > 120° (likely GPS noise)
                if abs(diff) <= 120:
                    total_rotation += diff
                    prev_heading = heading
                # For large jumps, keep the same heading (ignore the noise)

            return total_rotation, len(headings)

        # Method 3: Direction-consistent analysis
        def direction_consistent_analysis(headings):
            """Focus on the dominant turn direction"""

            changes = []
            for i in range(1, len(headings)):
                diff = headings[i] - headings[i-1]
                while diff > 180:
                    diff -= 360
                while diff < -180:
                    diff += 360

                if abs(diff) <= 120:  # Outlier rejection
                    changes.append(diff)

            if not changes:
                return 0, 0

            # Determine dominant direction
            positive_sum = sum(c for c in changes if c > 0)
            negative_sum = sum(c for c in changes if c < 0)

            if abs(negative_sum) > abs(positive_sum):
                # Left turn dominant
                dominant_changes = [c for c in changes if c < 0]
                direction = -1
            else:
                # Right turn dominant
                dominant_changes = [c for c in changes if c > 0]
                direction = 1

            total_dominant = sum(dominant_changes)
            return total_dominant, direction

        # Apply all methods
        raw_rotation, _ = progressive_heading_analysis(headings)
        smooth_rotation, _ = progressive_heading_analysis(smooth_hdg)
        dominant_rotation, direction = direction_consistent_analysis(headings)

        # Method 4: Full rotation detection
        def detect_full_rotations(headings):
            """Detect if we've made full 360° rotations"""

            if len(headings) < 10:  # Need sufficient data
                return 0

            # Look for heading wrapping patterns
            wraps = 0

            # Track crossings of 0°/360° line
            for i in range(1, len(headings)):
                prev = headings[i-1]
                curr = headings[i]

                # Detect 360° -> 0° crossing (positive rotation)
                if prev > 270 and curr < 90:
                    wraps += 1
                # Detect 0° -> 360° crossing (negative rotation)
                elif prev < 90 and curr > 270:
                    wraps -= 1

            return wraps

        full_rotations = detect_full_rotations(headings)

        # Method 5: Turn classification and validation
        def classify_and_validate(rotations_dict):
            """Classify turn type and validate with multiple methods"""

            methods_agreement = {}

            for method_name, rotation in rotations_dict.items():
                abs_rotation = abs(rotation)

                # Add full rotations if detected
                if full_rotations != 0:
                    abs_rotation += abs(full_rotations) * 360

                # Classify into standard turns
                if abs_rotation < 150:
                    category = 90
                elif abs_rotation < 350:
                    category = 270
                elif abs_rotation < 550:
                    category = 450
                elif abs_rotation < 750:
                    category = 630
                elif abs_rotation < 950:
                    category = 810
                else:
                    category = 990

                methods_agreement[method_name] = {
                    'raw_rotation': rotation,
                    'corrected_rotation': abs_rotation,
                    'category': category,
                    'confidence': self._calculate_rotation_confidence(abs_rotation, category)
                }

            return methods_agreement

        # Combine all methods
        rotation_methods = {
            'raw': raw_rotation,
            'smoothed': smooth_rotation,
            'dominant': dominant_rotation
        }

        classifications = classify_and_validate(rotation_methods)

        # Decision logic: pick the most confident method
        best_method = None
        best_confidence = 0

        for method_name, data in classifications.items():
            if data['confidence'] > best_confidence:
                best_confidence = data['confidence']
                best_method = method_name

        if best_method:
            result = classifications[best_method]
            final_rotation = result['corrected_rotation']
            intended_turn = result['category']
            confidence = result['confidence']

            # Apply direction
            if direction == -1:
                final_rotation = -final_rotation

            return final_rotation, intended_turn, confidence, best_method
        else:
            # Fallback to most conservative estimate
            return smooth_rotation, 270, 0.3, 'fallback'

    def _calculate_rotation_confidence(self, rotation, category):
        """Calculate confidence based on distance from standard turn"""
        distance = abs(rotation - category)
        max_distance = 90  # Maximum acceptable distance
        confidence = max(0, (max_distance - distance) / max_distance)
        return confidence

    def _get_rotation_legacy(self, df, flare_idx, max_gspeed_idx):
        """Legacy rotation calculation method (original algorithm)"""
        headings = df['heading'].astype(int)
        heading_unwrapped_deg = np.unwrap(np.deg2rad(headings), discont=np.deg2rad(20)) * 180/np.pi

        diff_per_record_deg = np.diff(heading_unwrapped_deg)

        seed = diff_per_record_deg[flare_idx : flare_idx+30]
        dom_sign = np.sign(np.nanmedian(seed))
        if dom_sign == 0:
            dom_sign = np.sign(np.nansum(seed)) or 1.0

        ref_heading = headings[flare_idx]
        rel = (headings - ref_heading + 360.0) % 360.0
        rel_unwrapped_deg = unwrap_deg(rel)

        net_change = 0
        last_heading = headings[flare_idx]
        passed_start = 0
        start_heading = headings[flare_idx] + (20 * dom_sign)
        last_less_than_start = True

        for heading in headings[flare_idx:max_gspeed_idx]:
            if (heading > last_heading and dom_sign < 0) or (heading < last_heading and dom_sign > 0):
                if abs(heading - last_heading) > 10:
                    if last_less_than_start:
                        passed_start += 1
                        last_less_than_start = False
            heading_change = heading - last_heading
            net_change += heading_change
            if (heading < start_heading - 20 and dom_sign < 0) or (heading > start_heading + 20 and dom_sign > 0):
                last_less_than_start = True
            last_heading = heading

        total_change = net_change + ((360.0 * dom_sign) * passed_start)

        # ENHANCED SELECTIVE FIXES FOR ROTATION ACCURACY
        magnitude = abs(total_change)

        # Fix for extreme over-counting (10,000+ degrees)
        if magnitude > 10000:
            magnitude = self._estimate_rotation_from_headings(df, flare_idx, max_gspeed_idx)

        # Fix for severe over-counting (2000-10000 degrees)
        elif magnitude > 2000:
            magnitude = self._estimate_rotation_from_headings(df, flare_idx, max_gspeed_idx)

        # Fix for moderate over-counting (800-2000 degrees)
        elif magnitude > 800:
            magnitude = magnitude - 360

        # Fix 1: Over-counting (750-800 should be ~450)
        elif magnitude > 750:
            magnitude = magnitude - 360

        # Fix 2: Under-counting for 270° that should be 630°
        elif 250 <= magnitude <= 300:
            if self._needs_extra_rotation(df, flare_idx, max_gspeed_idx):
                magnitude = magnitude + 360

        # Fix 3: Under-counting for ~90° that should be 450°
        elif 80 <= magnitude <= 120:
            if self._needs_450_correction(df, flare_idx, max_gspeed_idx):
                magnitude = magnitude + 360

        # Return with original direction
        return magnitude if total_change >= 0 else -magnitude

    def _needs_extra_rotation(self, df, flare_idx, max_gspeed_idx):
        """Helper method to detect 630° patterns"""
        turn_headings = df['heading'].astype(int)[flare_idx:max_gspeed_idx+1].values
        if len(turn_headings) < 3:
            return False

        total_turn = 0
        prev = turn_headings[0]
        for h in turn_headings[1:]:
            diff = h - prev
            if diff > 180: diff -= 360
            elif diff < -180: diff += 360
            total_turn += abs(diff)
            prev = h

        net_change = abs(turn_headings[-1] - turn_headings[0])
        return total_turn > 500 and net_change < 300

    def _needs_450_correction(self, df, flare_idx, max_gspeed_idx):
        """Helper method to detect 450° patterns"""
        turn_headings = df['heading'].astype(int)[flare_idx:max_gspeed_idx+1].values
        if len(turn_headings) < 3:
            return False

        total_turn = 0
        prev = turn_headings[0]
        for h in turn_headings[1:]:
            diff = h - prev
            if diff > 180: diff -= 360
            elif diff < -180: diff += 360
            total_turn += abs(diff)
            prev = h

        return total_turn > 350

    def _estimate_rotation_from_headings(self, df, flare_idx, max_gspeed_idx):
        """Estimate rotation directly from heading changes for extreme over-counting cases"""
        turn_headings = df['heading'].astype(int)[flare_idx:max_gspeed_idx+1].values

        if len(turn_headings) < 3:
            return 270  # Default reasonable estimate

        # Calculate net heading change
        start_heading = turn_headings[0]
        end_heading = turn_headings[-1]

        # Calculate shortest angular distance
        net_change = end_heading - start_heading
        if net_change > 180:
            net_change -= 360
        elif net_change < -180:
            net_change += 360

        # Calculate cumulative turn distance to detect full rotations
        total_turn = 0
        prev_heading = turn_headings[0]

        for heading in turn_headings[1:]:
            diff = heading - prev_heading
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            total_turn += abs(diff)
            prev_heading = heading

        # Estimate number of full rotations based on cumulative distance
        if total_turn > 600:
            # Likely includes at least one full rotation (360°)
            full_rotations = int(total_turn / 360)
            if full_rotations > 2:
                full_rotations = 1  # Cap at 1 full rotation for safety
            estimated_rotation = abs(net_change) + (full_rotations * 360)
        else:
            estimated_rotation = abs(net_change)

        # Cap at reasonable maximum (most swoops are 90°, 270°, 450°, or 630°)
        if estimated_rotation > 650:
            # For extreme cases, default to most common values
            if 200 < estimated_rotation < 350:
                estimated_rotation = 270
            elif 400 < estimated_rotation < 550:
                estimated_rotation = 450
            elif 550 < estimated_rotation < 700:
                estimated_rotation = 630
            else:
                estimated_rotation = 450  # Most common fallback

        return estimated_rotation

    def get_roll_out(self, df, max_vspeed_idx, max_gspeed_idx, landing_idx):
        """Find rollout start and end points"""
        roll_out_start_window = df.iloc[max_vspeed_idx:max_gspeed_idx+1]
        if roll_out_start_window.empty:
            roll_out_start = max_gspeed_idx
            plane_out_idx = self.find_plane_out(df, roll_out_start, landing_idx)
            return roll_out_start, plane_out_idx

        max_gspeed_heading = df.iloc[max_gspeed_idx]['heading']
        on_final_heading = (roll_out_start_window['heading'] - max_gspeed_heading).abs() <= 45
        if on_final_heading.any():
            roll_out_start = on_final_heading.index[on_final_heading][0]
        else:
            roll_out_start = roll_out_start_window.index[0]

        plane_out_idx = self.find_plane_out(df, roll_out_start, landing_idx)
        return roll_out_start, plane_out_idx

    def find_plane_out(self, df, roll_out_idx, landing_idx, near_zero=2.0, rise_thresh=2.0, rise_len=3, win_len=4, agl_range_max=1.0):
        """Find plane out point"""
        df_section = df.loc[roll_out_idx:landing_idx] if df.index.isin([roll_out_idx, landing_idx]).any() else df.iloc[roll_out_idx:landing_idx+1]
        df_section = df_section.reset_index(drop=False)

        if df_section.empty:
            return roll_out_idx

        vspeed = df_section['velD'].astype(float).abs().to_numpy()
        agl = df_section['AGL'].astype(float).to_numpy()
        n = len(df_section)

        # Primary: stable window
        if n >= win_len:
            near0_run = (
                pd.Series((vspeed <= near_zero).astype(int))
                .rolling(win_len).sum().to_numpy()
            )
            agl_roll_range = (
                    pd.Series(agl).rolling(win_len).max()
                    - pd.Series(agl).rolling(win_len).min()
            ).to_numpy()
            ok = (near0_run >= win_len) & (agl_roll_range <= agl_range_max)
            if ok.any():
                k = int(np.argmax(ok))
                start_row = max(0, k - win_len + 1)
                return int(df_section.iloc[start_row]['index'])

        # Fallback 1: low point then sustained rise
        vs = df_section['velD'].astype(float).to_numpy()
        low_pos = int(np.argmin(vs))
        low_val = float(vs[low_pos])
        tail = vs[low_pos:]
        if len(tail) >= rise_len:
            above = (tail >= (low_val + rise_thresh)).astype(int)
            if (pd.Series(above).rolling(rise_len).sum() >= rise_len).any():
                return int(df_section.iloc[low_pos]['index'])

        # Fallback 2: absolute minimum
        return int(df_section.iloc[low_pos]['index'])


# Convenience function for processing single files
def process_flysight_file(filepath, pilot, canopy=None):
    """Process a single FlySight file"""
    manager = FlightManager()
    return manager.process_file(filepath, pilot=pilot, canopy=canopy)


# Convenience function for batch processing
def process_directory(directory_path, pilot, canopy=None, file_pattern="*.csv"):
    """Process all FlySight files in a directory"""
    import glob
    manager = FlightManager()

    files = glob.glob(os.path.join(directory_path, file_pattern))
    files.extend(glob.glob(os.path.join(directory_path, file_pattern.upper())))

    results = []
    for filepath in files:
        try:
            flight = manager.process_file(filepath, pilot=pilot, canopy=canopy)
            results.append((filepath, flight, True))
        except Exception as e:
            results.append((filepath, None, False, str(e)))

    return results