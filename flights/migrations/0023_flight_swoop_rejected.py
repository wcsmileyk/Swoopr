from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flights', '0022_flight_max_g_force_avg_g_force_turn'),
    ]

    operations = [
        migrations.AddField(
            model_name='flight',
            name='swoop_rejected',
            field=models.BooleanField(
                default=False,
                help_text='User declared this is not a swoop landing — excluded from all analysis'
            ),
        ),
    ]
