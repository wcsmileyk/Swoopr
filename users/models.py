from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """Extended user profile for swoop pilots"""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')

    # Pilot information
    license_number = models.CharField(max_length=20, blank=True, help_text="Skydiving license number")
    uspa_number = models.CharField(max_length=20, blank=True, help_text="USPA member number")
    license_level = models.CharField(
        max_length=2,
        choices=[
            ('A', 'A License'),
            ('B', 'B License'),
            ('C', 'C License'),
            ('D', 'D License')
        ],
        blank=True
    )
    coach = models.BooleanField(default=False)
    affi = models.BooleanField(default=False)
    ti = models.BooleanField(default=False)

    # Jump experience
    total_jumps = models.IntegerField(null=True, blank=True)
    swoop_jumps = models.IntegerField(null=True, blank=True)
    exit_weight = models.FloatField(null=True, blank=True, help_text="Exit weight in lbs")

    # Contact and location
    home_dz = models.CharField(max_length=100, blank=True, help_text="Home drop zone")
    phone = models.CharField(max_length=20, blank=True)
    emergency_contact = models.CharField(max_length=100, blank=True)
    emergency_phone = models.CharField(max_length=20, blank=True)

    # Preferences
    units = models.CharField(
        max_length=10,
        choices=[('metric', 'Metric'), ('imperial', 'Imperial')],
        default='imperial'
    )
    timezone = models.CharField(max_length=50, default='UTC')
    public_profile = models.BooleanField(default=False, help_text="Allow others to see your stats")

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['license_level']),
            models.Index(fields=['home_dz']),
            models.Index(fields=['public_profile']),
        ]

    def __str__(self):
        return f"{self.user.username}'s Profile"

    @property
    def display_name(self):
        """Return user's preferred display name"""
        if self.user.first_name and self.user.last_name:
            return f"{self.user.first_name} {self.user.last_name}"
        return self.user.username

    @property
    def experience_level(self):
        """Calculate experience level based on jump numbers"""
        if not self.total_jumps:
            return "Unknown"
        elif self.total_jumps < 100:
            return "Beginner"
        elif self.total_jumps < 500:
            return "Intermediate"
        elif self.total_jumps < 1000:
            return "Advanced"
        else:
            return "Expert"

    @property
    def primary_canopy(self):
        """Return the user's primary/most recent canopy"""
        return self.canopies.filter(is_primary=True).first() or self.canopies.order_by('-created_at').first()


class Canopy(models.Model):
    """Canopy/parachute information"""

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='canopies')

    # Canopy details
    manufacturer = models.CharField(max_length=50, help_text="e.g., Icarus, Performance Designs")
    model = models.CharField(max_length=50, help_text="e.g., Sabre2, Katana")
    size = models.IntegerField(help_text="Size in square feet")
    year_manufactured = models.IntegerField(null=True, blank=True)

    # Configuration
    line_set = models.CharField(max_length=50, blank=True, help_text="Line set type if modified")
    modifications = models.TextField(blank=True, help_text="Any modifications made")

    # Status
    is_primary = models.BooleanField(default=False, help_text="Primary canopy for this user")
    is_active = models.BooleanField(default=True, help_text="Still in use")
    retired_date = models.DateField(null=True, blank=True)
    retirement_reason = models.CharField(max_length=100, blank=True)

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Canopies"
        indexes = [
            models.Index(fields=['user', 'is_primary']),
            models.Index(fields=['manufacturer', 'model']),
            models.Index(fields=['size']),
        ]
        # Ensure only one primary canopy per user
        constraints = [
            models.UniqueConstraint(
                fields=['user'],
                condition=models.Q(is_primary=True),
                name='unique_primary_canopy_per_user'
            )
        ]

    def __str__(self):
        return f"{self.manufacturer} {self.model} {self.size}sq ft"

    @property
    def wing_loading(self):
        """Calculate wing loading if user has exit weight"""
        if self.user.profile.exit_weight and self.size:
            return round(self.user.profile.exit_weight / self.size, 2)
        return None

    @property
    def display_name(self):
        """Short display name for the canopy"""
        return f"{self.model} {self.size}"

    def save(self, *args, **kwargs):
        # If this is being set as primary, unset other primaries for this user
        if self.is_primary:
            Canopy.objects.filter(user=self.user, is_primary=True).exclude(pk=self.pk).update(is_primary=False)
        super().save(*args, **kwargs)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Automatically create UserProfile when User is created"""
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """Save UserProfile when User is saved"""
    if hasattr(instance, 'profile'):
        instance.profile.save()