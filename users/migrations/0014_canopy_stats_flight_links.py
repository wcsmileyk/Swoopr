import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('flights', '0001_initial'),
        ('users', '0013_canopy_avg_horizontal_speed'),
    ]

    operations = [
        migrations.AddField(
            model_name='canopy',
            name='stats_max_horizontal_speed_flight',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='+',
                to='flights.flight',
            ),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_min_horizontal_speed_flight',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='+',
                to='flights.flight',
            ),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_max_vertical_speed_flight',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='+',
                to='flights.flight',
            ),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_min_vertical_speed_flight',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='+',
                to='flights.flight',
            ),
        ),
    ]
