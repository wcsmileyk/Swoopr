from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0006_userprofile_iad_sl'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='student_program',
            field=models.CharField(
                blank=True,
                choices=[
                    ('AFF', 'AFF'),
                    ('Tandem', 'Tandem'),
                    ('IAD_SL', 'IAD / Static Line'),
                    ('Coach', 'Coach'),
                ],
                help_text='Student jump program (visible when license level is Student)',
                max_length=10,
            ),
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='license_level',
            field=models.CharField(
                blank=True,
                choices=[
                    ('S', 'Student'),
                    ('A', 'A License'),
                    ('B', 'B License'),
                    ('C', 'C License'),
                    ('D', 'D License'),
                ],
                max_length=2,
            ),
        ),
    ]
