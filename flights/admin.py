from django.contrib import admin
from django.http import HttpResponse
from django.urls import path
from django.utils.html import format_html
from django.shortcuts import get_object_or_404
from django.db import models
from .models import Flight
import csv
from io import StringIO


class FlightAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'pilot', 'uploaded_at', 'is_swoop', 'analysis_successful',
        'data_incorrect', 'flare_detection_method', 'analysis_error_short',
        'download_csv_link'
    ]
    list_filter = [
        'analysis_successful', 'data_incorrect', 'is_swoop',
        'flare_detection_method', 'uploaded_at'
    ]
    search_fields = ['pilot__username', 'analysis_error', 'notes']
    readonly_fields = [
        'uploaded_at', 'analyzed_at', 'file_size', 'duration_seconds',
        'max_vertical_speed_ms', 'max_ground_speed_ms', 'turn_rotation'
    ]

    fieldsets = (
        ('Basic Info', {
            'fields': ('pilot', 'uploaded_at', 'file_size', 'duration_seconds')
        }),
        ('Analysis Status', {
            'fields': ('analysis_successful', 'analysis_error', 'analyzed_at', 'flare_detection_method')
        }),
        ('Swoop Data', {
            'fields': ('is_swoop', 'turn_rotation', 'turn_direction', 'max_vertical_speed_ms', 'max_ground_speed_ms'),
            'classes': ('collapse',)
        }),
        ('Flags & Notes', {
            'fields': ('data_incorrect', 'notes')
        })
    )

    def analysis_error_short(self, obj):
        """Show shortened analysis error"""
        if obj.analysis_error:
            return obj.analysis_error[:100] + '...' if len(obj.analysis_error) > 100 else obj.analysis_error
        return '-'
    analysis_error_short.short_description = 'Analysis Error'

    def download_csv_link(self, obj):
        """Add download CSV link"""
        if obj.id:
            return format_html(
                '<a href="{}download_csv/" class="button">Download CSV</a>',
                obj.id
            )
        return '-'
    download_csv_link.short_description = 'Download'
    download_csv_link.allow_tags = True

    def get_urls(self):
        """Add custom URL for CSV download"""
        urls = super().get_urls()
        custom_urls = [
            path(
                '<int:flight_id>/download_csv/',
                self.admin_site.admin_view(self.download_csv_view),
                name='flights_flight_download_csv'
            ),
        ]
        return custom_urls + urls

    def download_csv_view(self, request, flight_id):
        """Download flight data as CSV"""
        flight = get_object_or_404(Flight, id=flight_id)

        # Get GPS data from the flight
        gps_data = flight.get_gps_data()
        if not gps_data:
            # Return empty CSV if no data
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="flight_{flight_id}_no_data.csv"'
            return response

        # Create CSV content
        output = StringIO()
        writer = csv.writer(output)

        # Write header
        if gps_data:
            headers = list(gps_data[0].keys())
            writer.writerow(headers)

            # Write data rows
            for point in gps_data:
                writer.writerow([point.get(header, '') for header in headers])

        # Create response
        response = HttpResponse(output.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="flight_{flight_id}_{flight.pilot.username}.csv"'

        return response

    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        return super().get_queryset(request).select_related('pilot')


# Custom admin view for problematic flights
class ProblematicFlightAdmin(FlightAdmin):
    """Admin view focused on flights that need attention"""

    def get_queryset(self, request):
        """Show only failed or flagged flights"""
        qs = super().get_queryset(request)
        return qs.filter(
            models.Q(analysis_successful=False) |
            models.Q(data_incorrect=True)
        )

    def has_add_permission(self, request):
        """Don't allow adding from this view"""
        return False


# Register the admin classes
admin.site.register(Flight, FlightAdmin)

# Create a proxy model for the problematic flights view
class ProblematicFlight(Flight):
    class Meta:
        proxy = True
        verbose_name = "Problematic Flight"
        verbose_name_plural = "Problematic Flights"

admin.site.register(ProblematicFlight, ProblematicFlightAdmin)
