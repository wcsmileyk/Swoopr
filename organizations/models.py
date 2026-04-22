from django.conf import settings
from django.db import models


class Dropzone(models.Model):
    name = models.CharField(max_length=100, unique=True)
    city = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True, default='USA')
    icao = models.CharField(max_length=10, blank=True, help_text="Nearest airport ICAO code")

    class Meta:
        ordering = ['name']
        # Preserve the existing DB table name so no data migration is needed.
        db_table = 'logbook_dropzone'

    def __str__(self):
        parts = [self.name]
        if self.city and self.state:
            parts.append(f'{self.city}, {self.state}')
        elif self.city or self.state:
            parts.append(self.city or self.state)
        return ' — '.join(parts)


class DropzoneMembership(models.Model):
    """
    Links a user to a dropzone with a specific role.
    A user can have different roles at different dropzones.

    Role hierarchy (enforced in views/forms, not at DB level):
    - 'admin'  : Assigned by site admin only. Can manage all DZ functions,
                 grant staff roles, validate instructor ratings for their DZ,
                 and grant DropzoneAuthorizations.
    - 'staff'  : Assigned by a DZ admin. Can perform day-to-day DZ operations
                 as permitted by the admin.
    - 'member' : Basic affiliation — jumper is associated with this DZ.
    """
    ROLE_CHOICES = [
        ('admin', 'DZ Admin'),
        ('staff', 'Staff'),
        ('member', 'Member'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='dz_memberships'
    )
    dropzone = models.ForeignKey(
        Dropzone, on_delete=models.CASCADE, related_name='memberships'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='member')
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='memberships_granted',
        help_text='User who assigned this role. Admin roles must be granted by a site admin.'
    )
    granted_date = models.DateField(auto_now_add=True)
    active = models.BooleanField(default=True)

    class Meta:
        unique_together = [('user', 'dropzone')]
        ordering = ['dropzone', 'role', 'user']

    def __str__(self):
        return f'{self.user.username} — {self.get_role_display()} @ {self.dropzone.name}'


class Waiver(models.Model):
    """A liability waiver signed by a jumper at a specific dropzone."""

    dropzone = models.ForeignKey(Dropzone, on_delete=models.CASCADE, related_name='waivers')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='waivers'
    )
    signed_date = models.DateField()
    expiry_date = models.DateField(null=True, blank=True, help_text="Leave blank if waiver does not expire")
    waiver_version = models.CharField(max_length=20, blank=True, help_text="e.g., '2024-01', 'v3'")
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-signed_date']

    def __str__(self):
        return f'{self.user.username} — {self.dropzone.name} waiver ({self.signed_date})'

    @property
    def is_current(self):
        if self.expiry_date is None:
            return True
        from django.utils import timezone
        return self.expiry_date >= timezone.now().date()
