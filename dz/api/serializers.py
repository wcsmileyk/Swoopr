from rest_framework import serializers

from aircraft.models import Aircraft
from dz.models import DailyRoster, JumpRunConditions, Load, LoadSlot


class LoadSlotSerializer(serializers.ModelSerializer):
    display_name = serializers.SerializerMethodField()
    user_display = serializers.SerializerMethodField()

    class Meta:
        model = LoadSlot
        fields = [
            'id', 'display_name', 'user', 'user_display', 'guest_name',
            'role', 'jump_type', 'group_key',
            'exit_altitude', 'weight_lbs', 'media_status',
            'reservation', 'enrollment',
            'added_by', 'added_at',
        ]
        read_only_fields = ['added_by', 'added_at']

    def get_display_name(self, obj):
        return obj.display_name

    def get_user_display(self, obj):
        if obj.user:
            return obj.user.get_full_name() or obj.user.username
        return None


class LoadSerializer(serializers.ModelSerializer):
    slots = LoadSlotSerializer(many=True, read_only=True)
    slot_count = serializers.SerializerMethodField()
    aircraft_display = serializers.SerializerMethodField()

    class Meta:
        model = Load
        fields = [
            'id', 'load_number', 'date', 'status',
            'aircraft', 'aircraft_display',
            'call_time', 'took_off_at', 'landed_at',
            'exit_altitude', 'reserved_student_slots',
            'notes', 'slot_count', 'slots',
            'created_by', 'created_at',
        ]
        read_only_fields = [
            'load_number', 'status',
            'call_time', 'took_off_at', 'landed_at',
            'created_by', 'created_at',
        ]

    def get_slot_count(self, obj):
        return obj.slots.count()

    def get_aircraft_display(self, obj):
        return str(obj.aircraft) if obj.aircraft else None


class LoadCreateSerializer(serializers.Serializer):
    """Thin serializer for creating a load — delegates to load_service."""
    aircraft = serializers.PrimaryKeyRelatedField(
        queryset=Aircraft.objects.all(),
        required=False,
        allow_null=True,
    )
    date = serializers.DateField()


class LoadUpdateSerializer(serializers.ModelSerializer):
    """Fields staff can edit directly (not transitions)."""
    class Meta:
        model = Load
        fields = ['aircraft', 'exit_altitude', 'reserved_student_slots', 'notes']


class LoadSlotCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = LoadSlot
        fields = [
            'user', 'guest_name', 'role', 'jump_type', 'group_key',
            'exit_altitude', 'weight_lbs', 'media_status',
            'reservation', 'enrollment',
        ]

    def validate(self, data):
        if not data.get('user') and not data.get('guest_name'):
            raise serializers.ValidationError(
                'Either a platform user or a guest name is required.'
            )
        return data


class LoadSlotUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = LoadSlot
        fields = ['group_key', 'exit_altitude', 'weight_lbs', 'media_status', 'role', 'jump_type']


class JumpRunConditionsSerializer(serializers.ModelSerializer):
    set_by_display = serializers.SerializerMethodField()

    class Meta:
        model = JumpRunConditions
        fields = ['id', 'date', 'exit_altitude', 'spot_notes', 'set_by', 'set_by_display', 'set_at']
        read_only_fields = ['set_by', 'set_at']

    def get_set_by_display(self, obj):
        if obj.set_by:
            return obj.set_by.get_full_name() or obj.set_by.username
        return None


class JumpRunConditionsCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = JumpRunConditions
        fields = ['date', 'exit_altitude', 'spot_notes']


class RosterEntrySerializer(serializers.ModelSerializer):
    user_display = serializers.SerializerMethodField()
    ratings = serializers.SerializerMethodField()

    class Meta:
        model = DailyRoster
        fields = [
            'id', 'user', 'user_display', 'ratings',
            'availability_state', 'jump_count_today',
            'thirty_day_avg_at_checkin', 'date',
        ]

    def get_user_display(self, obj):
        return obj.user.get_full_name() or obj.user.username

    def get_ratings(self, obj):
        return list(obj.user.instructor_ratings.values_list('rating_type', flat=True))
