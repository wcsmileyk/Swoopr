from django.db import models


class Aircraft(models.Model):
    model = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100, blank=True)
    dropzone = models.ForeignKey(
        'organizations.Dropzone',
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='aircraft',
        help_text='Primary operating DZ. Null for shared/unassigned aircraft.',
    )
    tail_number = models.CharField(max_length=20, blank=True)
    jump_run_altitude = models.IntegerField(
        null=True, blank=True,
        help_text='Default exit altitude in feet for this aircraft.',
    )
    usable_slots = models.IntegerField(
        null=True, blank=True,
        help_text='Max jumpers per load.',
    )

    class Meta:
        ordering = ['manufacturer', 'model']
        verbose_name_plural = 'Aircraft'
        # Preserve the existing DB table name so no data migration is needed.
        db_table = 'logbook_aircraft'

    def __str__(self):
        parts = []
        if self.manufacturer:
            parts.append(self.manufacturer)
        parts.append(self.model)
        if self.tail_number:
            parts.append(f'({self.tail_number})')
        return ' '.join(parts)
