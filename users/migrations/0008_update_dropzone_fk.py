"""
Phase 0 structural migration.

Updates the UserProfile.home_dropzone FK to reference organizations.Dropzone
instead of logbook.Dropzone. No database operation is needed — the physical
FK still points to the same logbook_dropzone table.
"""
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0007_userprofile_student_program'),
        ('organizations', '0001_initial'),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.AlterField(
                    model_name='userprofile',
                    name='home_dropzone',
                    field=models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name='home_pilots',
                        to='organizations.dropzone',
                    ),
                ),
            ],
        ),
    ]
