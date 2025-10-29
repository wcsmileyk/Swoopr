from django.contrib import admin
from django.http import HttpResponse
from django.urls import path
from django.utils.html import format_html
from django.shortcuts import get_object_or_404
from django.db import models
from .models import Flight, CompetitionGate
import csv
from io import StringIO


class FlightAdmin(admin.ModelAdmin):
    list_display = [
        'id', 'pilot', 'uploaded_at', 'is_swoop', 'competition_gate', 'analysis_successful',
        'data_incorrect', 'flare_detection_method', 'total_flight_time',
        'analysis_error_short', 'download_csv_link'
    ]
    list_filter = [
        'analysis_successful', 'data_incorrect', 'is_swoop',
        'flare_detection_method', 'uploaded_at', 'competition_gate'
    ]
    search_fields = ['pilot__username', 'analysis_error', 'notes']
    readonly_fields = [
        'uploaded_at', 'analyzed_at', 'total_flight_time',
        'max_vertical_speed_ms', 'max_ground_speed_ms', 'turn_rotation',
        'gate_crossing_idx', 'gate_speed_mps', 'gate_altitude_agl', 'passed_between_gates'
    ]

    fieldsets = (
        ('Basic Info', {
            'fields': ('pilot', 'competition_gate', 'uploaded_at', 'total_flight_time', 'filename')
        }),
        ('Analysis Status', {
            'fields': ('analysis_successful', 'analysis_error', 'analyzed_at', 'flare_detection_method')
        }),
        ('Swoop Data', {
            'fields': ('is_swoop', 'turn_rotation', 'turn_direction', 'max_vertical_speed_ms', 'max_ground_speed_ms'),
            'classes': ('collapse',)
        }),
        ('Gate Metrics', {
            'fields': ('gate_crossing_idx', 'gate_speed_mps', 'gate_altitude_agl', 'passed_between_gates'),
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


class CompetitionGateAdmin(admin.ModelAdmin):
    """Admin interface for competition gates"""
    list_display = [
        'id', 'name', 'gate_type', 'is_parsed', 'created_by',
        'created_at', 'parse_action', 'view_map_link'
    ]
    list_filter = ['gate_type', 'is_parsed', 'created_at']
    search_fields = ['name', 'parse_error']
    readonly_fields = [
        'created_at', 'updated_at', 'is_parsed', 'parse_error',
        'gate_positions', 'course_config', 'center_lat', 'center_lon'
    ]

    fieldsets = (
        ('Basic Info', {
            'fields': ('name', 'gate_type', 'created_by', 'gate_file')
        }),
        ('Parse Status', {
            'fields': ('is_parsed', 'parse_error', 'created_at', 'updated_at')
        }),
        ('Parsed Data', {
            'fields': ('gate_positions', 'course_config', 'center_lat', 'center_lon'),
            'classes': ('collapse',)
        })
    )

    def parse_action(self, obj):
        """Add parse/reparse button"""
        if obj.id:
            if obj.is_parsed:
                return format_html(
                    '<a href="{}reparse/" class="button">Reparse</a>',
                    obj.id
                )
            else:
                return format_html(
                    '<a href="{}parse/" class="button">Parse Now</a>',
                    obj.id
                )
        return '-'
    parse_action.short_description = 'Actions'

    def view_map_link(self, obj):
        """Add link to view course on map"""
        if obj.id and obj.is_parsed:
            return format_html(
                '<a href="/flights/gates/{}/map/" class="button" target="_blank">View Map</a>',
                obj.id
            )
        return '-'
    view_map_link.short_description = 'Map'

    def get_urls(self):
        """Add custom URLs for parsing"""
        urls = super().get_urls()
        custom_urls = [
            path(
                '<int:gate_id>/parse/',
                self.admin_site.admin_view(self.parse_gate_view),
                name='flights_competitiongate_parse'
            ),
            path(
                '<int:gate_id>/reparse/',
                self.admin_site.admin_view(self.parse_gate_view),
                name='flights_competitiongate_reparse'
            ),
        ]
        return custom_urls + urls

    def parse_gate_view(self, request, gate_id):
        """Parse or reparse a gate file"""
        from django.contrib import messages
        from django.shortcuts import redirect

        gate = get_object_or_404(CompetitionGate, id=gate_id)

        if gate.parse_gate_file():
            messages.success(request, f"Successfully parsed gate file '{gate.name}'")
        else:
            messages.error(request, f"Failed to parse gate file: {gate.parse_error}")

        return redirect('admin:flights_competitiongate_change', gate_id)

    def save_model(self, request, obj, form, change):
        """Set created_by to current user if not set"""
        if not obj.created_by:
            obj.created_by = request.user
        super().save_model(request, obj, form, change)

        # Automatically parse on save if not already parsed
        if not obj.is_parsed and obj.gate_file:
            obj.parse_gate_file()


# Register the admin classes
admin.site.register(Flight, FlightAdmin)
admin.site.register(CompetitionGate, CompetitionGateAdmin)

# Create a proxy model for the problematic flights view
class ProblematicFlight(Flight):
    class Meta:
        proxy = True
        verbose_name = "Problematic Flight"
        verbose_name_plural = "Problematic Flights"

admin.site.register(ProblematicFlight, ProblematicFlightAdmin)
