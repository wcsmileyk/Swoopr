from django.contrib import admin

from .models import Dropzone, DropzoneMembership


class DropzoneMembershipInline(admin.TabularInline):
    model = DropzoneMembership
    extra = 0
    fields = ('user', 'role', 'granted_by', 'active')
    readonly_fields = ('granted_by',)


@admin.register(Dropzone)
class DropzoneAdmin(admin.ModelAdmin):
    list_display = ('name', 'city', 'state', 'country', 'icao')
    search_fields = ('name', 'city', 'state', 'icao')
    ordering = ('name',)
    inlines = [DropzoneMembershipInline]


@admin.register(DropzoneMembership)
class DropzoneMembershipAdmin(admin.ModelAdmin):
    list_display = ('user', 'dropzone', 'role', 'active', 'granted_by', 'granted_date')
    list_filter = ('role', 'active', 'dropzone')
    search_fields = ('user__username', 'user__first_name', 'user__last_name', 'dropzone__name')
    ordering = ('dropzone', 'role', 'user')
