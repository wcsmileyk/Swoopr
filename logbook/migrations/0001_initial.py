from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('flights', '0022_flight_max_g_force_avg_g_force_turn'),
        ('users', '0002_add_auto_public_flights'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Dropzone',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('city', models.CharField(blank=True, max_length=100)),
                ('state', models.CharField(blank=True, max_length=100)),
                ('country', models.CharField(blank=True, default='USA', max_length=100)),
                ('icao', models.CharField(blank=True, help_text='Nearest airport ICAO code', max_length=10)),
            ],
            options={'ordering': ['name']},
        ),
        migrations.CreateModel(
            name='Aircraft',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.CharField(max_length=100)),
                ('manufacturer', models.CharField(blank=True, max_length=100)),
            ],
            options={'ordering': ['manufacturer', 'model'], 'verbose_name_plural': 'Aircraft'},
        ),
        migrations.CreateModel(
            name='JumpType',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, unique=True)),
                ('working_jump', models.BooleanField(default=True, help_text='Counts as a training/working jump (vs fun/leisure)')),
            ],
            options={'ordering': ['name']},
        ),
        migrations.CreateModel(
            name='Jump',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('jump_number', models.PositiveIntegerField(blank=True, null=True)),
                ('date', models.DateField()),
                ('altitude', models.IntegerField(blank=True, help_text='Exit altitude in feet', null=True)),
                ('formation_size', models.IntegerField(blank=True, help_text='Number of people in formation', null=True)),
                ('swoop', models.BooleanField(default=False)),
                ('turn_rotation', models.IntegerField(blank=True, help_text='Manually entered turn rotation (degrees) if no GPS file', null=True)),
                ('notes', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='jumps',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('dropzone', models.ForeignKey(
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='jumps',
                    to='logbook.dropzone',
                )),
                ('aircraft', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='jumps',
                    to='logbook.aircraft',
                )),
                ('canopy', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='jumps',
                    to='users.canopy',
                )),
                ('jump_type', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='jumps',
                    to='logbook.jumptype',
                )),
                ('flight', models.OneToOneField(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='jump',
                    to='flights.flight',
                )),
            ],
            options={
                'ordering': ['user', 'date', 'created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='jump',
            index=models.Index(fields=['user', 'date'], name='logbook_jum_user_id_date_idx'),
        ),
        migrations.AddIndex(
            model_name='jump',
            index=models.Index(fields=['user', 'jump_number'], name='logbook_jum_user_id_num_idx'),
        ),
        migrations.AddIndex(
            model_name='jump',
            index=models.Index(fields=['user', 'swoop'], name='logbook_jum_user_id_swoop_idx'),
        ),
        migrations.AddIndex(
            model_name='jump',
            index=models.Index(fields=['flight'], name='logbook_jum_flight_idx'),
        ),
    ]
