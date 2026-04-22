from django.contrib import admin

from .models import Aircraft


@admin.register(Aircraft)
class AircraftAdmin(admin.ModelAdmin):
    list_display = ('manufacturer', 'model')
    search_fields = ('manufacturer', 'model')
    ordering = ('manufacturer', 'model')
