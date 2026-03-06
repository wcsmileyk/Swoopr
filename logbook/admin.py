from django.contrib import admin
from .models import Aircraft, Dropzone, Jump, JumpType


@admin.register(Dropzone)
class DropzoneAdmin(admin.ModelAdmin):
    list_display = ('name', 'city', 'state', 'country', 'icao')
    search_fields = ('name', 'city', 'state', 'icao')
    ordering = ('name',)


@admin.register(Aircraft)
class AircraftAdmin(admin.ModelAdmin):
    list_display = ('manufacturer', 'model')
    search_fields = ('manufacturer', 'model')
    ordering = ('manufacturer', 'model')


@admin.register(JumpType)
class JumpTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'working_jump')
    ordering = ('name',)


@admin.register(Jump)
class JumpAdmin(admin.ModelAdmin):
    list_display = ('jump_number', 'user', 'date', 'dropzone', 'jump_type', 'canopy', 'swoop', 'has_gps')
    list_filter = ('swoop', 'jump_type', 'dropzone', 'date')
    search_fields = ('user__username', 'dropzone__name', 'notes')
    raw_id_fields = ('flight',)
    ordering = ('user', '-date', '-jump_number')

    def has_gps(self, obj):
        return obj.flight_id is not None
    has_gps.boolean = True
    has_gps.short_description = 'GPS'
