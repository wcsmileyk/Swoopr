from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flights', '0023_flight_swoop_rejected'),
    ]

    operations = [
        migrations.AddField(
            model_name='flight',
            name='canopy_start_timestamp',
            field=models.FloatField(
                blank=True,
                null=True,
                help_text='Unix timestamp of canopy deployment + stabilization point (start of valid canopy data)',
            ),
        ),
    ]