import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aircraft', '0001_initial'),
        ('organizations', '0003_waiver'),
    ]

    operations = [
        migrations.AddField(
            model_name='aircraft',
            name='dropzone',
            field=models.ForeignKey(
                blank=True, null=True,
                help_text='Primary operating DZ. Null for shared/unassigned aircraft.',
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='aircraft',
                to='organizations.dropzone',
            ),
        ),
        migrations.AddField(
            model_name='aircraft',
            name='tail_number',
            field=models.CharField(blank=True, max_length=20),
        ),
        migrations.AddField(
            model_name='aircraft',
            name='jump_run_altitude',
            field=models.IntegerField(
                blank=True, null=True,
                help_text='Default exit altitude in feet for this aircraft.',
            ),
        ),
        migrations.AddField(
            model_name='aircraft',
            name='usable_slots',
            field=models.IntegerField(
                blank=True, null=True,
                help_text='Max jumpers per load.',
            ),
        ),
    ]
