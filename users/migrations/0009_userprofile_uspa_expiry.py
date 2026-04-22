from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0008_update_dropzone_fk'),
    ]

    operations = [
        migrations.AddField(
            model_name='userprofile',
            name='uspa_expiry',
            field=models.DateField(
                blank=True,
                null=True,
                help_text='USPA membership expiry date — governs all rating currency',
            ),
        ),
    ]
