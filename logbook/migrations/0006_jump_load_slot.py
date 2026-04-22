import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logbook', '0005_move_to_domain_apps'),
        ('dz', '0002_phase3_manifest'),
    ]

    operations = [
        migrations.AddField(
            model_name='jump',
            name='load_slot',
            field=models.OneToOneField(
                blank=True,
                help_text='DZ manifest slot this jump corresponds to.',
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='jump',
                to='dz.loadslot',
            ),
        ),
    ]
