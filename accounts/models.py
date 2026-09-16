from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom User model that maps to the existing auth_user table. Setting
    AUTH_USER_MODEL to this now lets us add fields and extend the user in
    future phases without a disruptive data migration.

    No new table is created — db_table points at the existing auth_user table
    and the migration uses SeparateDatabaseAndState to register the model in
    Django's state without touching the database.
    """

    class Meta:
        db_table = 'auth_user'


class Notification(models.Model):
    """
    In-app notification for a user. Not tied to a single domain app —
    reused across training sign-offs, rigging tickets, etc.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='notifications'
    )
    category = models.CharField(
        max_length=30,
        help_text="e.g. 'student_signoff', 'progression_ready', 'rig_ticket_update'",
    )
    title = models.CharField(max_length=200)
    body = models.TextField(blank=True)
    link_url = models.CharField(max_length=300, blank=True)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} — {self.title}'
