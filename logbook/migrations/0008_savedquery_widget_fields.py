from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logbook', '0007_savedquery'),
    ]

    operations = [
        migrations.AddField(
            model_name='savedquery',
            name='is_dashboard_widget',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='savedquery',
            name='is_swooper_widget',
            field=models.BooleanField(default=False),
        ),
    ]
