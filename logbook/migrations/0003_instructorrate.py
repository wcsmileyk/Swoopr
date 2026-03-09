from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logbook', '0002_rename_logbook_jum_user_id_date_idx_logbook_jum_user_id_b97fa1_idx_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='InstructorRate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('amount', models.DecimalField(decimal_places=2, help_text='Pay per jump in USD', max_digits=8)),
                ('jump_type', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='instructor_rates', to='logbook.jumptype')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='instructor_rates', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'unique_together': {('user', 'jump_type')},
            },
        ),
    ]
