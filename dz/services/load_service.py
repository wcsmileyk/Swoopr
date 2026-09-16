"""
Load service — business logic for manifest load lifecycle.

All state transitions on Load go through this module so that side effects
(call time computation, stats updates, BreakRequest wiring) are centralised
and not scattered across views.

Public API
----------
create_load(dz, date, created_by, aircraft=None)
call_load(load, called_by)
depart_load(load, user)        — on_call → in_air
land_load(load, user)          — in_air → landed  (also updates DailyAircraftStats and syncs draft Jumps)
uncall_load(load, user)        — on_call → building
compute_call_time(dz, aircraft, date)
set_call_time_override(load, user, new_time, reason='')
clear_call_time_override(load)
"""

import datetime

from django.db import transaction
from django.db.models import Count, Sum
from django.utils import timezone

# Rolling window for cross-day aircraft performance stats.
ROLLING_STATS_WINDOW_DAYS = 14

# DZ default call-time gap used only when there's no usable history at all.
DEFAULT_CALL_GAP = datetime.timedelta(minutes=30)

# Assumed per-jumper weight (lbs) baseline a load's climb performance is
# implicitly calibrated against; heavier-than-baseline loads get a climb
# time penalty.
BASELINE_WEIGHT_PER_JUMPER_LBS = 200
WEIGHT_PENALTY_PCT_PER_100_LBS_OVER = 0.03

# Simple weather hold buffers, added on top of the computed estimate.
WIND_HOLD_THRESHOLD_KTS = 20
WIND_HOLD_BUFFER = datetime.timedelta(minutes=10)
ACTIVE_HOLD_BUFFER = datetime.timedelta(minutes=20)


# ---------------------------------------------------------------------------
# Load creation
# ---------------------------------------------------------------------------

def create_load(dz, date, created_by, aircraft=None):
    """
    Create the next sequential load for this DZ on this date.
    Returns the new Load instance.
    """
    from dz.models import Load

    with transaction.atomic():
        existing = Load.objects.filter(dropzone=dz, date=date).select_for_update()
        next_number = (existing.order_by('-load_number').values_list('load_number', flat=True).first() or 0) + 1

        load = Load.objects.create(
            dropzone=dz,
            aircraft=aircraft,
            date=date,
            load_number=next_number,
            created_by=created_by,
            exit_altitude=aircraft.jump_run_altitude if aircraft else None,
        )
    return load


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def call_load(load, called_by):
    """
    Move load building → on_call.
    Uses a manual CallTimeOverride if one is set for this load; otherwise
    computes call_time via the cascade logic.
    """
    if hasattr(load, 'call_time_override'):
        call_time = load.call_time_override.overridden_call_time
    else:
        call_time = compute_call_time(load.dropzone, load.aircraft, load.date)
    load.call_time = call_time
    load.transition_to('on_call')   # saves
    return load


def uncall_load(load, user):
    """Move load on_call → building (reset call_time)."""
    load.call_time = None
    load.transition_to('building')
    return load


def depart_load(load, user):
    """Move load on_call → in_air."""
    load.took_off_at = timezone.now()
    load.transition_to('in_air')
    return load


@transaction.atomic
def land_load(load, user):
    """
    Move load in_air → landed.
    Updates DailyAircraftStats rolling averages if aircraft is set.
    Wires landed_at timestamp.
    Syncs completed slots into draft personal logbook entries.
    """
    load.landed_at = timezone.now()
    load.transition_to('landed')

    if load.aircraft and load.took_off_at and load.landed_at:
        _update_aircraft_stats(load)

    from logbook.services.manifest_sync import sync_slots_for_load
    sync_slots_for_load(load)

    return load


# ---------------------------------------------------------------------------
# Manual call-time override
# ---------------------------------------------------------------------------

def set_call_time_override(load, user, new_time, reason=''):
    """
    Set (or replace) a manual call-time override for a load.
    Audited — records who set it, when, and why.
    If the load is already on_call, updates its stored call_time immediately.
    """
    from dz.models import CallTimeOverride

    override, _ = CallTimeOverride.objects.update_or_create(
        load=load,
        defaults={
            'overridden_call_time': new_time,
            'reason': reason,
            'set_by': user,
        },
    )
    if load.status == 'on_call':
        load.call_time = new_time
        load.save(update_fields=['call_time'])
    return override


def clear_call_time_override(load):
    """Remove a manual override, reverting to the cascade on the next call."""
    from dz.models import CallTimeOverride
    CallTimeOverride.objects.filter(load=load).delete()


# ---------------------------------------------------------------------------
# Call time cascade
# ---------------------------------------------------------------------------

def compute_call_time(dz, aircraft, date):
    """
    Estimate when a newly called load will be ready to board.

    Cascade order:
    1. If an aircraft is currently in_air, estimate its landing (using rolling
       multi-day stats, adjusted for that in-air load's jumper weight) + avg
       turn time, plus a weather hold buffer if conditions call for one.
    2. If the most recent landed load today has stats, use last land time +
       avg turn time.
    3. Fall back to now + DZ default (30 minutes) — a "no data" estimate.

    Returns a datetime.
    """
    from dz.models import Load

    now = timezone.now()

    # 1. Aircraft currently in air
    if aircraft:
        in_air_load = (
            Load.objects
            .filter(dropzone=dz, date=date, aircraft=aircraft, status='in_air')
            .order_by('-took_off_at')
            .first()
        )
        if in_air_load and in_air_load.took_off_at:
            stats = _get_stats(aircraft, dz, date)
            altitude_min = _best_altitude_min(stats)
            turn_min = _best_turn_min(stats)
            if altitude_min and turn_min:
                total_weight, jumper_count = _load_weight_stats(in_air_load)
                flight_duration_min = _adjust_for_load_weight(
                    aircraft, altitude_min * 2, jumper_count, total_weight,
                )
                flight_duration = datetime.timedelta(minutes=flight_duration_min)
                turn_time = datetime.timedelta(minutes=turn_min)
                estimated_land = in_air_load.took_off_at + flight_duration
                estimate = max(now, estimated_land) + turn_time
                return estimate + _weather_delay(dz, date)

    # 2. Most recently landed load today
    last_landed = (
        Load.objects
        .filter(dropzone=dz, date=date, status='landed', landed_at__isnull=False)
        .order_by('-landed_at')
        .first()
    )
    if last_landed and last_landed.aircraft:
        stats = _get_stats(last_landed.aircraft, dz, date)
        turn_min = _best_turn_min(stats)
        if turn_min:
            turn_time = datetime.timedelta(minutes=turn_min)
            estimate = max(now, last_landed.landed_at) + turn_time
            return estimate + _weather_delay(dz, date)

    # 3. DZ default — no usable history at all.
    return now + DEFAULT_CALL_GAP + _weather_delay(dz, date)


def _best_altitude_min(stats):
    """Prefer the rolling multi-day average; fall back to same-day only."""
    if not stats:
        return None
    return stats.rolling_avg_altitude_min or stats.avg_altitude_min


def _best_turn_min(stats):
    if not stats:
        return None
    return stats.rolling_avg_turn_min or stats.avg_turn_min


def _load_weight_stats(load):
    """(total_weight_lbs, jumper_count) for a load's filled slots, ignoring slots with no weight entered."""
    agg = load.slots.aggregate(total_weight=Sum('weight_lbs'), count=Count('id'))
    return agg['total_weight'], agg['count']


def _adjust_for_load_weight(aircraft, base_climb_min, jumper_count, total_weight_lbs):
    """
    Scale a baseline climb-time estimate for a heavier-than-typical load.
    No-op unless the aircraft has a configured climb_rate_fpm and we have
    both a jumper count and a total weight to compare against baseline.
    """
    if not aircraft.climb_rate_fpm or not jumper_count or not total_weight_lbs:
        return base_climb_min

    baseline_weight = jumper_count * BASELINE_WEIGHT_PER_JUMPER_LBS
    over = total_weight_lbs - baseline_weight
    if over <= 0:
        return base_climb_min

    penalty = (over / 100.0) * WEIGHT_PENALTY_PCT_PER_100_LBS_OVER
    return base_climb_min * (1 + penalty)


def _weather_delay(dz, date):
    """
    Extra buffer to add to a call-time estimate based on the most recently
    logged JumpRunConditions for this DZ/date. Returns a timedelta (zero if
    no conditions logged, or conditions don't call for a hold).
    """
    from dz.models import JumpRunConditions

    latest = (
        JumpRunConditions.objects
        .filter(dropzone=dz, date=date)
        .order_by('-set_at')
        .first()
    )
    if not latest:
        return datetime.timedelta(0)
    if latest.hold_called:
        return ACTIVE_HOLD_BUFFER
    if latest.wind_speed_kts and latest.wind_speed_kts >= WIND_HOLD_THRESHOLD_KTS:
        return WIND_HOLD_BUFFER
    return datetime.timedelta(0)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _get_stats(aircraft, dz, date):
    from dz.models import DailyAircraftStats
    return DailyAircraftStats.objects.filter(aircraft=aircraft, dropzone=dz, date=date).first()


def _update_aircraft_stats(load):
    """
    Recompute same-day averages for this aircraft/dropzone/date, plus a
    trailing ROLLING_STATS_WINDOW_DAYS-day rolling average across dates, and
    capture jumper-count/weight sample data for this day.
    Called inside land_load's transaction.
    """
    from dz.models import DailyAircraftStats, Load

    completed = list(
        Load.objects
        .filter(
            aircraft=load.aircraft,
            dropzone=load.dropzone,
            date=load.date,
            status='landed',
            took_off_at__isnull=False,
            landed_at__isnull=False,
        )
        .order_by('load_number')
        .values('load_number', 'took_off_at', 'landed_at', 'call_time')
    )

    if not completed:
        return

    # Altitude time: took_off_at → landed_at divided by 2 approximates climb time
    altitude_mins = [
        (c['landed_at'] - c['took_off_at']).total_seconds() / 60 / 2
        for c in completed
    ]

    # Turn time: landed_at[n] → took_off_at[n+1]
    turn_mins = []
    for i in range(len(completed) - 1):
        gap = (completed[i + 1]['took_off_at'] - completed[i]['landed_at']).total_seconds() / 60
        if 0 < gap < 120:   # sanity check — ignore gaps over 2 hours
            turn_mins.append(gap)

    avg_altitude = sum(altitude_mins) / len(altitude_mins) if altitude_mins else None
    avg_turn = sum(turn_mins) / len(turn_mins) if turn_mins else None

    weight_total, jumper_count = _load_weight_stats(load)

    stats, _ = DailyAircraftStats.objects.update_or_create(
        aircraft=load.aircraft,
        dropzone=load.dropzone,
        date=load.date,
        defaults={
            'load_count': len(completed),
            'avg_altitude_min': round(avg_altitude, 1) if avg_altitude else None,
            'avg_turn_min': round(avg_turn, 1) if avg_turn else None,
            'sample_weight_lbs_total': weight_total,
            'sample_jumper_count': jumper_count,
        },
    )

    _update_rolling_stats(load.aircraft, load.dropzone, load.date, stats)


def _update_rolling_stats(aircraft, dz, date, today_stats):
    """
    Recompute rolling_avg_altitude_min/rolling_avg_turn_min as a simple
    trailing-window average of same-day averages across the last
    ROLLING_STATS_WINDOW_DAYS days (including today).
    """
    from dz.models import DailyAircraftStats

    window_start = date - datetime.timedelta(days=ROLLING_STATS_WINDOW_DAYS - 1)
    window_days = (
        DailyAircraftStats.objects
        .filter(aircraft=aircraft, dropzone=dz, date__gte=window_start, date__lte=date)
        .exclude(pk=today_stats.pk)
        .values_list('avg_altitude_min', 'avg_turn_min')
    )

    altitude_vals = [today_stats.avg_altitude_min] if today_stats.avg_altitude_min else []
    turn_vals = [today_stats.avg_turn_min] if today_stats.avg_turn_min else []
    for alt, turn in window_days:
        if alt:
            altitude_vals.append(alt)
        if turn:
            turn_vals.append(turn)

    today_stats.rolling_avg_altitude_min = round(sum(altitude_vals) / len(altitude_vals), 1) if altitude_vals else None
    today_stats.rolling_avg_turn_min = round(sum(turn_vals) / len(turn_vals), 1) if turn_vals else None
    today_stats.save(update_fields=['rolling_avg_altitude_min', 'rolling_avg_turn_min'])
