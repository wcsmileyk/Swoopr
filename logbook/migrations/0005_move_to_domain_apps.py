"""
Phase 0 structural migration.

Removes Dropzone, Aircraft, StudentSignoff, and PendingInstructorRequest from
the logbook migration state and updates the Jump FK references to point at
their new app labels. No database operations are performed — all tables keep
their existing names and the physical schema is untouched.

Depends on:
  - organizations/0001 (owns Dropzone state going forward)
  - aircraft/0001      (owns Aircraft state going forward)
  - training/0001      (owns StudentSignoff + PendingInstructorRequest state)
  - logbook/0004       (last logbook migration before this one)
"""
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logbook', '0004_jump_student_category_jump_student_jump_method_and_more'),
        ('organizations', '0001_initial'),
        ('aircraft', '0001_initial'),
        ('training', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                # Remove models from logbook state — they now live in their
                # respective domain apps.
                migrations.DeleteModel(name='Dropzone'),
                migrations.DeleteModel(name='Aircraft'),
                migrations.DeleteModel(name='StudentSignoff'),
                migrations.DeleteModel(name='PendingInstructorRequest'),

                # Update Jump FKs to reference the new app labels.
                migrations.AlterField(
                    model_name='jump',
                    name='dropzone',
                    field=models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name='jumps',
                        to='organizations.dropzone',
                    ),
                ),
                migrations.AlterField(
                    model_name='jump',
                    name='aircraft',
                    field=models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name='jumps',
                        to='aircraft.aircraft',
                    ),
                ),
            ],
        ),
    ]
