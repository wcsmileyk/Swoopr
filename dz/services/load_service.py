"""
Load service — business logic for manifest load lifecycle.

All state transitions on Load go through this module so that side effects
(call time computation, stats updates, BreakRequest wiring) are centralised
and not scattered across views.

Public API
----------
create_load(dz, date, created_by, aircraft=None)
call_load(load, called_by)
send_load(load, user)          — building → on_call shortcut with time computation
depart_load(load, user)        — on_call → in_air
land_load(load, user)          — in_air → landed  (also updates DailyAircraftStats)
uncall_load(load, user)        — on_call → building
compute_call_time(dz, aircraft, date)
"""

import datetime

from django.db import transaction
from django.utils import timezone


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
    Computes and stores call_time via the cascade logic.
    """
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
    """
    load.landed_at = timezone.now()
    load.transition_to('landed')

    if load.aircraft and load.took_off_at and load.landed_at:
        _update_aircraft_stats(load)

    return load


# ---------------------------------------------------------------------------
# Call time cascade
# ---------------------------------------------------------------------------

def compute_call_time(dz, aircraft, date):
    """
    Estimate when a newly called load will be ready to board.

    Cascade order:
    1. If an aircraft is currently in_air, estimate its landing + avg turn time.
    2. If the most recent landed load has stats, use last land time + avg turn time.
    3. Fall back to now + DZ default (30 minutes).

    Returns a datetime.
    """
    from dz.models import DailyAircraftStats, Load

    now = timezone.now()
    default_gap = datetime.timedelta(minutes=30)

    # 1. Aircraft currently in air
    if aircraft:
        in_air_load = (
            Load.objects
            .filter(dropzone=dz, date=date, aircraft=aircraft, status='in_air')
            .order_by('-took_off_at')
            .first()
        )
        if in_air_load and in_air_load.took_off_at:
            stats = _get_stats(aircraft, date)
            if stats and stats.avg_altitude_min and stats.avg_turn_min:
                flight_duration = datetime.timedelta(minutes=stats.avg_altitude_min * 2)
                turn_time = datetime.timedelta(minutes=stats.avg_turn_min)
                estimated_land = in_air_load.took_off_at + flight_duration
                return max(now, estimated_land) + turn_time

    # 2. Most recently landed load today
    last_landed = (
        Load.objects
        .filter(dropzone=dz, date=date, status='landed', landed_at__isnull=False)
        .order_by('-landed_at')
        .first()
    )
    if last_landed and last_landed.aircraft:
        stats = _get_stats(last_landed.aircraft, date)
        if stats and stats.avg_turn_min:
            turn_time = datetime.timedelta(minutes=stats.avg_turn_min)
            return max(now, last_landed.landed_at) + turn_time

    # 3. DZ default
    return now + default_gap


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _get_stats(aircraft, date):
    from dz.models import DailyAircraftStats
    return DailyAircraftStats.objects.filter(aircraft=aircraft, date=date).first()


def _update_aircraft_stats(load):
    """
    Recompute rolling averages for this aircraft on this date
    using all completed loads (those with both took_off_at and landed_at).
    Called inside land_load's transaction.
    """
    from dz.models import DailyAircraftStats, Load

    completed = list(
        Load.objects
        .filter(
            aircraft=load.aircraft,
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

    DailyAircraftStats.objects.update_or_create(
        aircraft=load.aircraft,
        date=load.date,
        defaults={
            'load_count': len(completed),
            'avg_altitude_min': round(avg_altitude, 1) if avg_altitude else None,
            'avg_turn_min': round(avg_turn, 1) if avg_turn else None,
        },
    )
