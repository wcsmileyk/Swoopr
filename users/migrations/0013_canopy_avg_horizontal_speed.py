from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0012_canopy_stats_fields'),
    ]

    operations = [
        migrations.AddField(
            model_name='canopy',
            name='stats_avg_horizontal_speed_mph',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
