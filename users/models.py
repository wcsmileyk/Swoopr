from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    """Extended user profile for swoop pilots"""

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='profile')

    # Pilot information
    license_number = models.CharField(max_length=20, blank=True, help_text="Skydiving license number")
    uspa_number = models.CharField(max_length=20, blank=True, help_text="USPA member number")
    uspa_expiry = models.DateField(null=True, blank=True, help_text="USPA membership expiry date — governs all rating currency")
    uspa_official = models.BooleanField(default=False, help_text="Can validate and assign USPA ratings platform-wide. Assigned by site admin only.")
    license_level = models.CharField(
        max_length=2,
        choices=[
            ('S', 'Student'),
            ('A', 'A License'),
            ('B', 'B License'),
            ('C', 'C License'),
            ('D', 'D License'),
        ],
        blank=True
    )
    student_program = models.CharField(
        max_length=10,
        choices=[
            ('AFF', 'AFF'),
            ('Tandem', 'Tandem'),
            ('IAD_SL', 'IAD / Static Line'),
            ('Coach', 'Coach'),
        ],
        blank=True,
        help_text='Student jump program (visible when license level is Student)'
    )
    coach = models.BooleanField(default=False)
    affi = models.BooleanField(default=False)
    ti = models.BooleanField(default=False)
    iad_sl = models.BooleanField(default=False, help_text='IAD / Static Line Instructor rating')
    videographer = models.BooleanField(default=False)
    pro_rating = models.BooleanField(default=False, help_text='USPA Pro rating')

    # Jump experience
    total_jumps = models.IntegerField(null=True, blank=True)
    swoop_jumps = models.IntegerField(null=True, blank=True)
    exit_weight = models.FloatField(null=True, blank=True, help_text="Exit weight in lbs")

    # Contact and location
    home_dz = models.CharField(max_length=100, blank=True, help_text="Home drop zone (legacy text)")
    home_dropzone = models.ForeignKey(
        'organizations.Dropzone', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='home_pilots'
    )
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
    auto_public_flights = models.BooleanField(default=False, help_text="Automatically make all uploaded flights public")

    # Logbook settings
    jump_number_offset = models.PositiveIntegerField(
        default=1,
        help_text='Starting jump number. All jumps are renumbered from this value.'
    )

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

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='canopies')

    # Canopy details
    manufacturer = models.CharField(max_length=50, help_text="e.g., Icarus, Performance Designs")
    model = models.CharField(max_length=50, help_text="e.g., Sabre2, Katana")
    size = models.IntegerField(help_text="Size in square feet")
    year_manufactured = models.IntegerField(null=True, blank=True)

    # Canopy type flags
    elliptical = models.BooleanField(default=False, help_text='Highly elliptical planform')
    cross_braced = models.BooleanField(default=False, help_text='Cross-braced (e.g. Velocity, VK)')
    schuemann = models.BooleanField(default=False, help_text='Schümann planform (e.g. Crossfire, Katana)')

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


class Rig(models.Model):
    """A jumper's container/harness system."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='rigs')

    # Container
    manufacturer = models.CharField(max_length=50, help_text="e.g., Javelin, Vector, Mirage, Infinity")
    model = models.CharField(max_length=50, blank=True)
    serial_number = models.CharField(max_length=50, blank=True)
    dom = models.DateField(null=True, blank=True, help_text="Date of manufacture")

    # Reserve
    reserve_manufacturer = models.CharField(max_length=50, blank=True)
    reserve_model = models.CharField(max_length=50, blank=True)
    reserve_size = models.IntegerField(null=True, blank=True, help_text="Size in square feet")
    reserve_dom = models.DateField(null=True, blank=True, help_text="Reserve date of manufacture")
    reserve_repack_date = models.DateField(null=True, blank=True, help_text="Last reserve repack date (180-day cycle)")

    # AAD
    aad_manufacturer = models.CharField(max_length=50, blank=True, help_text="e.g., Cypres, VIGIL, MARS")
    aad_serial_number = models.CharField(max_length=50, blank=True)
    aad_service_date = models.DateField(null=True, blank=True, help_text="Next AAD service due date")

    # Status
    is_primary = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_primary', '-created_at']

    def __str__(self):
        parts = [self.manufacturer]
        if self.model:
            parts.append(self.model)
        if self.serial_number:
            parts.append(f'S/N {self.serial_number}')
        return ' '.join(parts)

    def save(self, *args, **kwargs):
        if self.is_primary:
            Rig.objects.filter(user=self.user, is_primary=True).exclude(pk=self.pk).update(is_primary=False)
        super().save(*args, **kwargs)




