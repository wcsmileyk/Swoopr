"""
Straight-line canopy performance stats computed from GPS tracks.

Canopy phase detection:
  - velocity_down > 11 ft/s (3.353 m/s) — excludes level airplane flight on jump run
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
MIN_CANOPY_VSPEED_MS = 11 * 0.3048     # 11 ft/s -> m/s — excludes level airplane flight
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
        (df['velocity_down'] > MIN_CANOPY_VSPEED_MS) &
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


def _compute_glide_ratios(df):
    """
    Compute per-segment glide ratios: total horizontal distance / total altitude lost.

    Per-interval ratios are biased high because GPS altitude noise is large relative to
    the altitude change in a single 0.2s sample. Using segment totals makes the noise on
    the two endpoint measurements negligible relative to several metres of true descent.

    Splits the straight-line DataFrame into segments at time gaps > MAX_GAP_S, then for
    each segment computes:
        glide_ratio = sum(h_speed * dt) / (alt_start - alt_end)

    The vAcc gate (same MAD-based threshold as flight_manager) is applied to segment
    endpoints — those altitude readings drive the denominator.
    """
    if len(df) < 2:
        return []

    timestamps = df['timestamp'].to_numpy(dtype=float)
    alt_agl = df['altitude_agl'].to_numpy(dtype=float)
    h_spd = df['h_speed'].to_numpy(dtype=float)

    # vAcc gate — same MAD approach as flight_manager
    v_acc_raw = df['v_acc'].to_numpy() if 'v_acc' in df.columns else np.full(len(df), np.nan)
    v_acc = pd.to_numeric(pd.Series(v_acc_raw), errors='coerce').to_numpy(dtype=float)
    valid = ~np.isnan(v_acc)
    if valid.sum() > 2:
        med = float(np.median(v_acc[valid]))
        mad = float(np.median(np.abs(v_acc[valid] - med))) or 1.0
        vacc_thr = med + 2.5 * mad
        vacc_ok = np.where(valid, v_acc <= vacc_thr, True)
    else:
        vacc_ok = np.ones(len(df), dtype=bool)

    # Identify segment boundaries (gaps in the straight-line data)
    seg_starts = [0]
    for i in range(1, len(df)):
        if timestamps[i] - timestamps[i - 1] > MAX_GAP_S:
            seg_starts.append(i)
    seg_starts.append(len(df))  # sentinel

    ratios = []
    for k in range(len(seg_starts) - 1):
        s = seg_starts[k]
        e = seg_starts[k + 1]  # exclusive

        if e - s < 2:
            continue
        if not vacc_ok[s] or not vacc_ok[e - 1]:
            continue

        alt_loss = alt_agl[s] - alt_agl[e - 1]
        if alt_loss <= 0:
            continue

        h_dist = float(np.sum(h_spd[s:e - 1] * np.diff(timestamps[s:e])))
        if h_dist > 0:
            ratios.append(h_dist / alt_loss)

    return ratios


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
        all_glide.extend(_compute_glide_ratios(straight))

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
