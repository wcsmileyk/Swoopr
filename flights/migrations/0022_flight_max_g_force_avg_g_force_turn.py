from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flights', '0021_flight_competition_gate_flight_gate_altitude_agl_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='flight',
            name='max_g_force',
            field=models.FloatField(blank=True, help_text='Peak g-force load during swoop turn', null=True),
        ),
        migrations.AddField(
            model_name='flight',
            name='avg_g_force_turn',
            field=models.FloatField(blank=True, help_text='Average g-force during swoop turn phase', null=True),
        ),
    ]
