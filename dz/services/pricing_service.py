"""
Pricing service — dynamic fare calculation for reservations.

Priority order (highest wins; min_price_cents is always enforced as a floor):
  1. PricingOverride for the specific date — bypasses tiers and time surcharges
  2. Active demand tier + any matching TimeSlotPricing surcharge
  3. PricingConfig.base_price_cents (no tiers configured or none match)

Public API
----------
get_current_price(dz, jump_type, date, time_slot=None) -> int | None
apply_pricing(reservation)                             -> None  (sets quoted_price_cents, no save)
"""

import datetime


def get_current_price(dz, jump_type, date, time_slot=None):
    """
    Return the price in cents for a new reservation at dz/jump_type/date/time_slot.
    Returns None if no active PricingConfig exists for this combination.
    """
    from dz.models import PricingConfig, Reservation

    try:
        config = (
            PricingConfig.objects
            .prefetch_related('tiers', 'time_slots', 'overrides')
            .get(dropzone=dz, jump_type=jump_type, is_active=True)
        )
    except PricingConfig.DoesNotExist:
        return None

    # 1. Date override (highest priority — manual hard-set for this date)
    override = config.overrides.filter(date=date).first()
    if override:
        return max(override.price_cents, config.min_price_cents)

    # 2. Demand count — pending+confirmed+checked_in on this date
    demand_count = Reservation.objects.filter(
        dropzone=dz,
        jump_type=jump_type,
        date=date,
        status__in=('pending', 'confirmed', 'checked_in'),
    ).count()

    # Active tier = highest min_sold that is <= demand_count
    active_tiers = list(
        config.tiers.filter(min_sold__lte=demand_count).order_by('-min_sold')
    )
    base_price = active_tiers[0].price_cents if active_tiers else config.base_price_cents

    # 3. Time slot surcharges — stack all matching windows
    surcharge = 0
    if time_slot is not None:
        day_of_week = date.weekday()
        for ts in config.time_slots.filter(is_active=True):
            if ts.time_from <= time_slot <= ts.time_to and ts.applies_on(day_of_week):
                surcharge += ts.surcharge_cents

    price = base_price + surcharge
    return max(price, config.min_price_cents)


def apply_pricing(reservation):
    """
    Compute and set quoted_price_cents on a Reservation instance.
    Does NOT save — caller is responsible for calling .save() or .full_clean().
    """
    price = get_current_price(
        reservation.dropzone,
        reservation.jump_type,
        reservation.date,
        reservation.time_slot,
    )
    reservation.quoted_price_cents = price
