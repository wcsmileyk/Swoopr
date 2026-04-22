from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0009_userprofile_uspa_expiry'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='uspa_official',
            field=models.BooleanField(
                default=False,
                help_text='Can validate and assign USPA ratings platform-wide. Assigned by site admin only.',
            ),
        ),
    ]
