from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_userprofile_and_canopy_updates'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='jump_number_offset',
            field=models.PositiveIntegerField(
                default=1,
                help_text='Starting jump number. All jumps are renumbered from this value.'
            ),
        ),
    ]
