from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logbook', '0001_initial'),
        ('users', '0002_add_auto_public_flights'),
    ]

    operations = [
        # UserProfile: instructor/rating flags
        migrations.AddField(
            model_name='userprofile',
            name='videographer',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='userprofile',
            name='pro_rating',
            field=models.BooleanField(default=False, help_text='USPA Pro rating'),
        ),
        # UserProfile: home dropzone FK (alongside legacy home_dz char field)
        migrations.AddField(
            model_name='userprofile',
            name='home_dropzone',
            field=models.ForeignKey(
                blank=True, null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name='home_pilots',
                to='logbook.dropzone',
            ),
        ),
        # Canopy: type flags
        migrations.AddField(
            model_name='canopy',
            name='elliptical',
            field=models.BooleanField(default=False, help_text='Highly elliptical planform'),
        ),
        migrations.AddField(
            model_name='canopy',
            name='cross_braced',
            field=models.BooleanField(default=False, help_text='Cross-braced (e.g. Velocity, VK)'),
        ),
        migrations.AddField(
            model_name='canopy',
            name='schuemann',
            field=models.BooleanField(default=False, help_text='Schümann planform (e.g. Crossfire, Katana)'),
        ),
    ]
