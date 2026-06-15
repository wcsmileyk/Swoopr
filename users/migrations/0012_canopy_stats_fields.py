from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0011_rig'),
    ]

    operations = [
        migrations.AddField(
            model_name='canopy',
            name='stats_avg_glide_ratio',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_avg_vertical_speed_mph',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_max_vertical_speed_mph',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_min_vertical_speed_mph',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_max_horizontal_speed_mph',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_min_horizontal_speed_mph',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_flight_count',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='canopy',
            name='stats_updated_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]