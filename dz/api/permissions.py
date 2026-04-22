from rest_framework.permissions import BasePermission

from organizations.models import DropzoneMembership


class IsDZStaff(BasePermission):
    """Allow site staff or active admin/staff members of the target DZ."""

    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        if request.user.is_staff:
            return True
        dz_id = view.kwargs.get('dz_id')
        if not dz_id:
            return False
        return DropzoneMembership.objects.filter(
            user=request.user,
            dropzone_id=dz_id,
            role__in=['admin', 'staff'],
            active=True,
        ).exists()


class IsDZAdmin(BasePermission):
    """Allow site staff or active admin members of the target DZ."""

    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        if request.user.is_staff:
            return True
        dz_id = view.kwargs.get('dz_id')
        if not dz_id:
            return False
        return DropzoneMembership.objects.filter(
            user=request.user,
            dropzone_id=dz_id,
            role='admin',
            active=True,
        ).exists()
