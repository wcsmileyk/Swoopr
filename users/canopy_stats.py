"""
Straight-line canopy performance stats computed from GPS tracks.

Canopy phase detection:
  - velocity_down > 0 (descending, not climbing with aircraft)
  - velocity_down < 70 mph (31.3 m/s) — post-deployment; XRW loads may have high descent
  - h_speed < 100 mph (44.7 m/s) — excludes freefall horizontal component
  - altitude_agl >= 1000ft (304.8m) — above swoop initiation zone

Straight-line: heading stays within 10 degrees for 2+ continuous seconds,
no internal data gap > 1 second.
"""

import logging

import numpy as np
import pandas as pd
from django.utils import timezone

logger = logging.getLogger('users.canopy_stats')

MAX_CANOPY_VSPEED_MS = 70 * 0.44704    # 70 mph -> m/s
MAX_CANOPY_HSPEED_MS = 100 * 0.44704   # 100 mph -> m/s
MIN_ALT_AGL_M = 1000 * 0.3048          # 1000ft -> meters
HEADING_THRESHOLD_DEG = 10.0
MIN_SEGMENT_DURATION_S = 2.0
MAX_GAP_S = 1.0
MPS_TO_MPH = 1.0 / 0.44704


def _circular_range(angles):
    """Minimum arc in degrees spanning all angles (handles 0/360 wrap)."""
    if len(angles) <= 1:
        return 0.0
    s = sorted(angles)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    gaps.append(360 - s[-1] + s[0])
    return 360 - max(gaps)


def _find_straight_segments(times, headings):
    """Return list of (start_idx, end_idx) inclusive for qualifying straight segments."""
    n = len(times)
    segments = []
    i = 0
    while i < n:
        j = i
        head_window = [headings[j]]
        while j + 1 < n:
            if times[j + 1] - times[j] > MAX_GAP_S:
                break
            candidate = head_window + [headings[j + 1]]
            if _circular_range(candidate) <= HEADING_THRESHOLD_DEG:
                head_window.append(headings[j + 1])
                j += 1
            else:
                break
        if times[j] - times[i] >= MIN_SEGMENT_DURATION_S:
            segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


def _extract_straight_line_canopy_points(flight):
    """
    Returns a DataFrame of qualifying canopy straight-line points from one flight, or None.
    """
    gps_data = flight.get_gps_data()
    if not gps_data:
        return None

    df = pd.DataFrame(gps_data)

    required = {'timestamp', 'altitude_agl', 'velocity_down', 'velocity_north', 'velocity_east', 'heading'}
    if not required.issubset(df.columns):
        return None

    for col in ['altitude_agl', 'velocity_down', 'velocity_north', 'velocity_east', 'heading']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=list(required)).reset_index(drop=True)

    df['h_speed'] = np.sqrt(df['velocity_north'] ** 2 + df['velocity_east'] ** 2)

    canopy = df[
        (df['altitude_agl'] >= MIN_ALT_AGL_M) &
        (df['velocity_down'] > 0) &
        (df['velocity_down'] < MAX_CANOPY_VSPEED_MS) &
        (df['h_speed'] < MAX_CANOPY_HSPEED_MS)
    ].copy().reset_index(drop=True)

    if len(canopy) < 4:
        return None

    segments = _find_straight_segments(
        canopy['timestamp'].tolist(),
        canopy['heading'].tolist(),
    )
    if not segments:
        return None

    straight_idx = set()
    for start, end in segments:
        straight_idx.update(range(start, end + 1))

    return canopy.iloc[sorted(straight_idx)].copy()


def update_canopy_stats(canopy):
    """
    Recompute and save straight-line canopy performance stats from all valid GPS flights.
    Called automatically after any flight for this canopy is saved.
    """
    from flights.models import Flight

    flights_qs = (
        Flight.objects
        .filter(canopy=canopy, analysis_successful=True)
        .exclude(swoop_rejected=True)
        .exclude(data_incorrect=True)
    )

    all_vspeed = []
    all_hspeed = []
    all_glide = []
    flight_count = 0

    for flight in flights_qs:
        if not flight.gps_data_compressed:
            continue
        try:
            straight = _extract_straight_line_canopy_points(flight)
        except Exception as exc:
            logger.warning("canopy stats: skipping flight %s: %s", flight.id, exc)
            continue
        if straight is None or straight.empty:
            continue

        flight_count += 1
        all_vspeed.extend(straight['velocity_down'].tolist())
        all_hspeed.extend(straight['h_speed'].tolist())

        glide_rows = straight[straight['velocity_down'] > 0.5]
        if not glide_rows.empty:
            all_glide.extend((glide_rows['h_speed'] / glide_rows['velocity_down']).tolist())

    if all_vspeed:
        canopy.stats_avg_vertical_speed_mph = round(float(np.mean(all_vspeed)) * MPS_TO_MPH, 2)
        canopy.stats_max_vertical_speed_mph = round(float(np.max(all_vspeed)) * MPS_TO_MPH, 2)
        canopy.stats_min_vertical_speed_mph = round(float(np.min(all_vspeed)) * MPS_TO_MPH, 2)
        canopy.stats_avg_horizontal_speed_mph = round(float(np.mean(all_hspeed)) * MPS_TO_MPH, 2)
        canopy.stats_max_horizontal_speed_mph = round(float(np.max(all_hspeed)) * MPS_TO_MPH, 2)
        canopy.stats_min_horizontal_speed_mph = round(float(np.min(all_hspeed)) * MPS_TO_MPH, 2)
        canopy.stats_avg_glide_ratio = round(float(np.mean(all_glide)), 2) if all_glide else None
    else:
        canopy.stats_avg_vertical_speed_mph = None
        canopy.stats_max_vertical_speed_mph = None
        canopy.stats_min_vertical_speed_mph = None
        canopy.stats_avg_horizontal_speed_mph = None
        canopy.stats_max_horizontal_speed_mph = None
        canopy.stats_min_horizontal_speed_mph = None
        canopy.stats_avg_glide_ratio = None

    canopy.stats_flight_count = flight_count
    canopy.stats_updated_at = timezone.now()
    canopy.save(update_fields=[
        'stats_avg_glide_ratio',
        'stats_avg_horizontal_speed_mph',
        'stats_avg_vertical_speed_mph',
        'stats_max_vertical_speed_mph',
        'stats_min_vertical_speed_mph',
        'stats_max_horizontal_speed_mph',
        'stats_min_horizontal_speed_mph',
        'stats_flight_count',
        'stats_updated_at',
    ])

    return flight_count
