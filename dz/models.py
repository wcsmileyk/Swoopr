from django.conf import settings
from django.db import models
from django.utils import timezone

from organizations.models import Dropzone


# ---------------------------------------------------------------------------
# Phase 3 — Manifest
# ---------------------------------------------------------------------------

class JumpRunConditions(models.Model):
    """
    Append-only log of jump run condition changes for a DZ.
    Conditions are DZ-wide and persist across loads — only logged when something
    changes (wind shift, different spot, altitude change, etc.).
    Query the most recent entry for a date to get current conditions.
    """
    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='jump_run_log')
    date = models.DateField()
    exit_altitude = models.IntegerField(help_text='Exit altitude in feet.')
    spot_notes = models.TextField(
        blank=True,
        help_text='Spot description, heading, landmarks, etc.',
    )
    set_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, related_name='jump_run_entries',
    )
    set_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-set_at']

    def __str__(self):
        return f'{self.dropzone.name} jump run — {self.exit_altitude}ft — {self.date}'


class Load(models.Model):
    """
    A single jump load. Status flows through a one-way state machine.
    Business logic (call time computation, stats update) lives in dz/services/load_service.py.
    """
    STATUS_CHOICES = [
        ('building',  'Building'),
        ('on_call',   'On Call'),
        ('in_air',    'In Air'),
        ('landed',    'Landed'),
    ]
    VALID_TRANSITIONS = {
        'building': ['on_call'],
        'on_call':  ['in_air', 'building'],   # can un-call back to building
        'in_air':   ['landed'],
        'landed':   [],
    }

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='loads')
    aircraft = models.ForeignKey(
        'aircraft.Aircraft', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='loads',
    )
    date = models.DateField()
    load_number = models.PositiveIntegerField(
        help_text='Sequential load number for this DZ on this date.',
    )
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='building')

    # Timestamps set by the service layer on each transition
    call_time = models.DateTimeField(
        null=True, blank=True,
        help_text='When this load was called (moved to on_call).',
    )
    took_off_at = models.DateTimeField(null=True, blank=True)
    landed_at = models.DateTimeField(null=True, blank=True)

    exit_altitude = models.IntegerField(
        null=True, blank=True,
        help_text='Planned exit altitude in feet. Defaults to aircraft default if set.',
    )
    reserved_student_slots = models.IntegerField(
        default=0,
        help_text='Number of slots held for students not yet assigned.',
    )
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, related_name='loads_created',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['date', 'load_number']
        unique_together = [('dropzone', 'date', 'load_number')]

    def __str__(self):
        return f'Load {self.load_number} — {self.dropzone.name} {self.date} [{self.status}]'

    def transition_to(self, new_status):
        allowed = self.VALID_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition load from '{self.status}' to '{new_status}'. "
                f"Allowed: {allowed or 'none (terminal state)'}."
            )
        self.status = new_status
        self.save()

    @property
    def is_active(self):
        return self.status in ('building', 'on_call', 'in_air')

    @property
    def slot_count(self):
        return self.slots.count()

    @property
    def jumper_count(self):
        """Filled slots — excludes reserved_student_slots."""
        return self.slots.count()


class LoadSlot(models.Model):
    """
    One jumper's assignment on a load.
    group_key groups paired slots (tandem pair, AFF group, multi-video party, etc.).
    Slots with the same group_key on the same load are treated as a party.
    """
    ROLE_CHOICES = [
        ('student',      'Student'),
        ('instructor',   'Instructor'),
        ('fun_jumper',   'Fun Jumper'),
        ('videographer', 'Videographer'),
        ('hop_n_pop',    'Hop & Pop'),
    ]
    JUMP_TYPE_CHOICES = [
        ('tandem',    'Tandem'),
        ('aff',       'AFF'),
        ('coach',     'Coach'),
        ('fun_jump',  'Fun Jump'),
        ('hop_n_pop', 'Hop & Pop'),
        ('tracking',  'Tracking'),
        ('wingsuit',  'Wingsuit'),
    ]
    MEDIA_STATUS_CHOICES = [
        ('none',      'None'),
        ('requested', 'Requested'),
        ('confirmed', 'Confirmed'),
        ('delivered', 'Delivered'),
    ]

    load = models.ForeignKey(Load, on_delete=models.CASCADE, related_name='slots')

    # Jumper identity — platform user or guest name
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='load_slots',
    )
    guest_name = models.CharField(max_length=100, blank=True)

    role = models.CharField(max_length=15, choices=ROLE_CHOICES)
    jump_type = models.CharField(max_length=15, choices=JUMP_TYPE_CHOICES)

    # Grouping — same group_key = same party on this load
    group_key = models.PositiveIntegerField(
        null=True, blank=True,
        help_text='Slots with the same group_key are a paired party (tandem pair, AFF group, etc.).',
    )

    exit_altitude = models.IntegerField(
        null=True, blank=True,
        help_text='Exit altitude for this slot. Overrides load default if set.',
    )
    weight_lbs = models.IntegerField(null=True, blank=True)
    media_status = models.CharField(max_length=10, choices=MEDIA_STATUS_CHOICES, default='none')

    # Optional back-references to upstream records
    reservation = models.ForeignKey(
        'dz.Reservation', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='load_slots',
        help_text='Reservation this slot was created from, if any.',
    )
    enrollment = models.ForeignKey(
        'training.StudentEnrollment', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='load_slots',
        help_text='Student enrollment this slot is associated with, if any.',
    )

    added_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='load_slots_added',
    )
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['load', 'group_key', 'role']

    def __str__(self):
        name = self.guest_name or (self.user.get_full_name() if self.user else 'Unknown')
        return f'{name} — {self.get_role_display()} on Load {self.load.load_number}'

    @property
    def display_name(self):
        if self.guest_name:
            return self.guest_name
        if self.user:
            return self.user.get_full_name() or self.user.username
        return 'Unknown'


class DailyAircraftStats(models.Model):
    """
    Rolling performance averages per aircraft per day.
    Updated by the service layer each time a Load lands.
    Used by call time cascade logic to estimate when an aircraft will be ready.
    """
    aircraft = models.ForeignKey(
        'aircraft.Aircraft', on_delete=models.CASCADE, related_name='daily_stats',
    )
    date = models.DateField()
    load_count = models.IntegerField(default=0)
    avg_altitude_min = models.FloatField(
        null=True, blank=True,
        help_text='Average minutes from takeoff to exit altitude.',
    )
    avg_turn_min = models.FloatField(
        null=True, blank=True,
        help_text='Average minutes from landing to next takeoff.',
    )

    class Meta:
        unique_together = [('aircraft', 'date')]
        ordering = ['-date', 'aircraft']

    def __str__(self):
        return f'{self.aircraft} stats — {self.date}'


class OperatingDay(models.Model):
    """
    Per-DZ schedule configuration for a given day of the week.
    slot_capacities stores max bookings per jump type per hour:
    e.g. {"09:00": {"tandem": 2, "aff": 1}, "10:00": {"tandem": 2}}
    A flat dict {"tandem": 8, "aff": 2} is also accepted for daily totals.
    """
    DAY_CHOICES = [
        (0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), (3, 'Thursday'),
        (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday'),
    ]

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='operating_days')
    day_of_week = models.IntegerField(choices=DAY_CHOICES)
    is_open = models.BooleanField(default=True)
    first_load_time = models.TimeField(null=True, blank=True)
    reservation_cutoff_hours = models.IntegerField(
        default=24,
        help_text='How many hours before the jump time reservations close.',
    )
    slot_capacities = models.JSONField(
        default=dict, blank=True,
        help_text='Max bookings per jump type (and optionally per hour).',
    )
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = [('dropzone', 'day_of_week')]
        ordering = ['dropzone', 'day_of_week']

    def __str__(self):
        return f'{self.dropzone.name} — {self.get_day_of_week_display()}'


class CheckIn(models.Model):
    """Physical arrival log for a jumper or instructor for a single day."""
    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='check_ins')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='check_ins',
    )
    date = models.DateField()
    checked_in_at = models.DateTimeField(default=timezone.now)
    checked_in_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='check_ins_processed',
        help_text='Staff member who processed the check-in. Null if self-check-in.',
    )
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = [('dropzone', 'user', 'date')]
        ordering = ['-date', '-checked_in_at']

    def __str__(self):
        return f'{self.user.get_full_name() or self.user.username} @ {self.dropzone.name} on {self.date}'


class DailyRoster(models.Model):
    """Instructor availability state for a specific day at a DZ."""
    AVAILABILITY_CHOICES = [
        ('available', 'Available'),
        ('limited', 'Limited'),
        ('unavailable', 'Unavailable'),
        ('on_break', 'On Break'),
    ]

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='daily_rosters')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='roster_entries',
    )
    date = models.DateField()
    availability_state = models.CharField(
        max_length=15, choices=AVAILABILITY_CHOICES, default='available',
    )
    jump_count_today = models.IntegerField(default=0)
    thirty_day_avg_at_checkin = models.FloatField(
        null=True, blank=True,
        help_text='Instructor 30-day jump average captured at check-in time.',
    )
    check_in = models.OneToOneField(
        CheckIn, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='roster_entry',
    )
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = [('dropzone', 'user', 'date')]
        ordering = ['-date', 'user__last_name']

    def __str__(self):
        return f'{self.user.get_full_name() or self.user.username} roster — {self.dropzone.name} {self.date}'


class BreakRequest(models.Model):
    """Instructor break request and approval workflow."""
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('approved', 'Approved'),
        ('denied', 'Denied'),
    ]

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='break_requests')
    instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='break_requests',
    )
    date = models.DateField()
    roster = models.ForeignKey(
        DailyRoster, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='break_requests',
    )
    reason = models.TextField(blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='break_requests_reviewed',
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    # load = models.ForeignKey('dz.Load', on_delete=models.SET_NULL, null=True, blank=True)
    # ^ stubbed — wired up in Phase 3 when Load model is introduced

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Break request — {self.instructor.get_full_name() or self.instructor.username} @ {self.dropzone.name} {self.date}'

    def approve(self, reviewed_by):
        self.status = 'approved'
        self.reviewed_by = reviewed_by
        self.reviewed_at = timezone.now()
        self.save()

    def deny(self, reviewed_by):
        self.status = 'denied'
        self.reviewed_by = reviewed_by
        self.reviewed_at = timezone.now()
        self.save()


class Reservation(models.Model):
    """
    Customer reservation for a jump experience.
    Most reservations are guests without a platform account — user FK is optional.
    The state machine is enforced via transition_to().
    """
    TYPE_CHOICES = [
        ('tandem', 'Tandem'),
        ('aff', 'AFF'),
        ('coach', 'Coach Jump'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('checked_in', 'Checked In'),
        ('jumped', 'Jumped'),
        ('cancelled', 'Cancelled'),
        ('no_show', 'No Show'),
    ]
    SOURCE_CHOICES = [
        ('online', 'Online'),
        ('phone', 'Phone'),
        ('walk_in', 'Walk-in'),
    ]
    VALID_TRANSITIONS = {
        'pending':    ['confirmed', 'cancelled'],
        'confirmed':  ['checked_in', 'cancelled', 'no_show'],
        'checked_in': ['jumped', 'cancelled'],
        'jumped':     [],
        'cancelled':  [],
        'no_show':    [],
    }

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='reservations')
    jump_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    date = models.DateField()
    time_slot = models.TimeField(null=True, blank=True, help_text='Requested jump time slot.')

    # Guest path — user is optional, guest fields cover non-platform customers
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='reservations',
        help_text='Platform account if the customer has one.',
    )
    guest_name = models.CharField(max_length=100, blank=True)
    guest_email = models.EmailField(blank=True)
    guest_phone = models.CharField(max_length=20, blank=True)
    weight_lbs = models.IntegerField(
        null=True, blank=True,
        help_text='Customer weight in lbs. Used for tandem weight limit checks.',
    )

    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default='pending')
    assigned_instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='instructor_reservations',
    )
    source = models.CharField(max_length=10, choices=SOURCE_CHOICES, default='online')
    notes = models.TextField(blank=True)

    # Pricing
    quoted_price_cents = models.IntegerField(
        null=True, blank=True,
        help_text='Price shown to the customer when booking.',
    )
    charged_price_cents = models.IntegerField(
        null=True, blank=True,
        help_text='Final amount actually charged.',
    )
    add_ons = models.JSONField(
        default=dict, blank=True,
        help_text='e.g. {"video": true, "photos": true}',
    )

    # Token for guest access to their booking confirmation (no account required)
    confirmation_token = models.CharField(max_length=64, blank=True, null=True, unique=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['date', 'time_slot']

    def __str__(self):
        name = self.guest_name or (self.user.get_full_name() if self.user else 'Unknown')
        return f'{self.get_jump_type_display()} — {name} @ {self.dropzone.name} {self.date}'

    @property
    def display_name(self):
        if self.guest_name:
            return self.guest_name
        if self.user:
            return self.user.get_full_name() or self.user.username
        return 'Unknown'

    @property
    def display_email(self):
        return self.guest_email or (self.user.email if self.user else '')

    def transition_to(self, new_status):
        allowed = self.VALID_TRANSITIONS.get(self.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Cannot transition reservation from '{self.status}' to '{new_status}'. "
                f"Allowed: {allowed or 'none (terminal state)'}."
            )
        self.status = new_status
        self.save()

    @property
    def quoted_price_display(self):
        if self.quoted_price_cents is None:
            return None
        return f'${self.quoted_price_cents / 100:.2f}'

    @property
    def charged_price_display(self):
        if self.charged_price_cents is None:
            return None
        return f'${self.charged_price_cents / 100:.2f}'


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class PricingConfig(models.Model):
    """
    Per-DZ, per-jump-type pricing configuration.
    base_price_cents is the fallback when no demand tier matches.
    min_price_cents is an absolute floor — the price will never go below it.
    """
    JUMP_TYPE_CHOICES = [
        ('tandem', 'Tandem'),
        ('aff', 'AFF'),
        ('coach', 'Coach Jump'),
    ]

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='pricing_configs')
    jump_type = models.CharField(max_length=10, choices=JUMP_TYPE_CHOICES)
    base_price_cents = models.IntegerField(help_text='Default price when no demand tier applies.')
    min_price_cents = models.IntegerField(default=0, help_text='Absolute floor — never charge less.')
    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = [('dropzone', 'jump_type')]
        ordering = ['dropzone', 'jump_type']

    def __str__(self):
        return f'{self.dropzone.name} — {self.get_jump_type_display()} pricing'

    @property
    def base_price_display(self):
        return f'${self.base_price_cents / 100:.2f}'

    @property
    def min_price_display(self):
        return f'${self.min_price_cents / 100:.2f}'


class PricingTier(models.Model):
    """
    Demand-based fare bucket. Active tier = highest min_sold ≤ current booking count.
    e.g. min_sold=0 → $199, min_sold=5 → $229, min_sold=10 → $259.
    """
    config = models.ForeignKey(PricingConfig, on_delete=models.CASCADE, related_name='tiers')
    min_sold = models.IntegerField(
        help_text='Minimum confirmed+pending bookings on this date to activate this tier.',
    )
    price_cents = models.IntegerField()
    label = models.CharField(max_length=50, blank=True, help_text='e.g. "Early Bird", "Standard", "Peak"')

    class Meta:
        ordering = ['config', 'min_sold']
        unique_together = [('config', 'min_sold')]

    def __str__(self):
        label = f' ({self.label})' if self.label else ''
        return f'{self.config} — from {self.min_sold} sold: ${self.price_cents / 100:.2f}{label}'

    @property
    def price_display(self):
        return f'${self.price_cents / 100:.2f}'


class TimeSlotPricing(models.Model):
    """
    Recurring time-window surcharge that stacks on the demand-tier price.
    e.g. sunset jumps (17:00-20:00) add $50 regardless of demand tier.
    days_of_week is a list of integers [0-6]; empty list means every day.
    """
    config = models.ForeignKey(PricingConfig, on_delete=models.CASCADE, related_name='time_slots')
    label = models.CharField(max_length=100, help_text='e.g. "Sunset Jump", "Sunrise Special"')
    time_from = models.TimeField()
    time_to = models.TimeField()
    surcharge_cents = models.IntegerField(help_text='Added on top of demand-tier price.')
    days_of_week = models.JSONField(
        default=list, blank=True,
        help_text='Day integers [0=Mon … 6=Sun] this applies to. Empty = every day.',
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['config', 'time_from']

    def __str__(self):
        return f'{self.label} +${self.surcharge_cents / 100:.2f} ({self.time_from.strftime("%H:%M")}–{self.time_to.strftime("%H:%M")})'

    def applies_on(self, day_of_week):
        return not self.days_of_week or day_of_week in self.days_of_week

    @property
    def surcharge_display(self):
        return f'+${self.surcharge_cents / 100:.2f}'


class PricingOverride(models.Model):
    """
    Manual date-specific price override. Takes priority over all tiers and surcharges.
    Use for holidays, special events, or manual corrections.
    """
    config = models.ForeignKey(PricingConfig, on_delete=models.CASCADE, related_name='overrides')
    date = models.DateField()
    price_cents = models.IntegerField()
    reason = models.CharField(max_length=200, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='pricing_overrides',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('config', 'date')]
        ordering = ['-date']

    def __str__(self):
        return f'{self.config} override {self.date}: ${self.price_cents / 100:.2f}'

    @property
    def price_display(self):
        return f'${self.price_cents / 100:.2f}'


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------

class Payment(models.Model):
    """
    Provider-agnostic payment record linked to a Reservation.
    Real providers are not implemented — StubProvider is used until a real integration is added.
    Swap providers by changing the provider field and wiring up the provider class in payment_service.py.
    """
    PROVIDER_CHOICES = [
        ('stub',   'Stub (no-op)'),
        ('stripe', 'Stripe'),
        ('square', 'Square'),
        ('cash',   'Cash'),
    ]
    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('authorized', 'Authorized'),
        ('captured',   'Captured'),
        ('refunded',   'Refunded'),
        ('failed',     'Failed'),
        ('voided',     'Voided'),
    ]

    reservation = models.OneToOneField(
        'dz.Reservation', on_delete=models.CASCADE, related_name='payment',
    )
    provider = models.CharField(max_length=10, choices=PROVIDER_CHOICES, default='stub')
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default='pending')
    amount_cents = models.IntegerField()
    currency = models.CharField(max_length=3, default='USD')
    provider_transaction_id = models.CharField(max_length=255, blank=True)
    provider_data = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Payment {self.pk} — {self.reservation} — ${self.amount_cents / 100:.2f} [{self.status}]'

    @property
    def amount_display(self):
        return f'${self.amount_cents / 100:.2f}'
