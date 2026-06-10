from django.contrib import admin

from .models import (
    BreakRequest, CheckIn, DailyAircraftStats, DailyRoster,
    JumpRunConditions, Load, LoadSlot, OperatingDay, Payment,
    PricingConfig, PricingOverride, PricingTier, Reservation, TimeSlotPricing,
)


@admin.register(OperatingDay)
class OperatingDayAdmin(admin.ModelAdmin):
    list_display = ['dropzone', 'get_day_of_week_display', 'is_open', 'first_load_time', 'reservation_cutoff_hours']
    list_filter = ['dropzone', 'is_open', 'day_of_week']


@admin.register(CheckIn)
class CheckInAdmin(admin.ModelAdmin):
    list_display = ['user', 'dropzone', 'date', 'checked_in_at', 'checked_in_by']
    list_filter = ['dropzone', 'date']
    search_fields = ['user__username', 'user__first_name', 'user__last_name']
    date_hierarchy = 'date'


@admin.register(DailyRoster)
class DailyRosterAdmin(admin.ModelAdmin):
    list_display = ['user', 'dropzone', 'date', 'availability_state', 'jump_count_today', 'thirty_day_avg_at_checkin']
    list_filter = ['dropzone', 'date', 'availability_state']
    search_fields = ['user__username', 'user__first_name', 'user__last_name']
    date_hierarchy = 'date'


@admin.register(BreakRequest)
class BreakRequestAdmin(admin.ModelAdmin):
    list_display = ['instructor', 'dropzone', 'date', 'status', 'reviewed_by', 'reviewed_at']
    list_filter = ['dropzone', 'status', 'date']
    search_fields = ['instructor__username', 'instructor__first_name', 'instructor__last_name']
    date_hierarchy = 'date'


@admin.register(Reservation)
class ReservationAdmin(admin.ModelAdmin):
    list_display = ['display_name', 'dropzone', 'jump_type', 'date', 'time_slot', 'status', 'source', 'assigned_instructor']
    list_filter = ['dropzone', 'jump_type', 'status', 'source', 'date']
    search_fields = ['guest_name', 'guest_email', 'user__username', 'user__first_name', 'user__last_name']
    date_hierarchy = 'date'

    def display_name(self, obj):
        return obj.display_name
    display_name.short_description = 'Customer'


@admin.register(JumpRunConditions)
class JumpRunConditionsAdmin(admin.ModelAdmin):
    list_display = ['dropzone', 'date', 'exit_altitude', 'set_by', 'set_at']
    list_filter = ['dropzone', 'date']
    date_hierarchy = 'date'


class LoadSlotInline(admin.TabularInline):
    model = LoadSlot
    extra = 0
    fields = ['display_name_col', 'role', 'jump_type', 'group_key', 'exit_altitude', 'weight_lbs', 'media_status']
    readonly_fields = ['display_name_col', 'added_at']

    def display_name_col(self, obj):
        return obj.display_name
    display_name_col.short_description = 'Jumper'


@admin.register(Load)
class LoadAdmin(admin.ModelAdmin):
    list_display = ['load_number', 'dropzone', 'date', 'aircraft', 'status', 'slot_count', 'call_time', 'landed_at']
    list_filter = ['dropzone', 'status', 'date']
    date_hierarchy = 'date'
    inlines = [LoadSlotInline]

    def slot_count(self, obj):
        return obj.slots.count()
    slot_count.short_description = 'Slots'


@admin.register(LoadSlot)
class LoadSlotAdmin(admin.ModelAdmin):
    list_display = ['display_name_col', 'load', 'role', 'jump_type', 'group_key', 'media_status']
    list_filter = ['role', 'jump_type', 'media_status']
    search_fields = ['guest_name', 'user__username', 'user__first_name', 'user__last_name']

    def display_name_col(self, obj):
        return obj.display_name
    display_name_col.short_description = 'Jumper'


@admin.register(DailyAircraftStats)
class DailyAircraftStatsAdmin(admin.ModelAdmin):
    list_display = ['aircraft', 'date', 'load_count', 'avg_altitude_min', 'avg_turn_min']
    list_filter = ['aircraft', 'date']
    date_hierarchy = 'date'


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

class PricingTierInline(admin.TabularInline):
    model = PricingTier
    extra = 1
    fields = ['min_sold', 'price_cents', 'label']


class TimeSlotPricingInline(admin.TabularInline):
    model = TimeSlotPricing
    extra = 1
    fields = ['label', 'time_from', 'time_to', 'surcharge_cents', 'days_of_week', 'is_active']


class PricingOverrideInline(admin.TabularInline):
    model = PricingOverride
    extra = 1
    fields = ['date', 'price_cents', 'reason']
    readonly_fields = ['created_by', 'created_at']


@admin.register(PricingConfig)
class PricingConfigAdmin(admin.ModelAdmin):
    list_display = ['dropzone', 'jump_type', 'base_price_cents', 'min_price_cents', 'is_active']
    list_filter = ['dropzone', 'jump_type', 'is_active']
    inlines = [PricingTierInline, TimeSlotPricingInline, PricingOverrideInline]


@admin.register(Payment)
class PaymentAdmin(admin.ModelAdmin):
    list_display = ['pk', 'reservation', 'provider', 'status', 'amount_cents', 'created_at']
    list_filter = ['provider', 'status']
    search_fields = ['reservation__guest_name', 'reservation__guest_email', 'provider_transaction_id']
    readonly_fields = ['created_at', 'updated_at']
