from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Registers aircraft.Aircraft in the migration state.
    The logbook_aircraft table already exists — no database operations needed.
    logbook/0005 will subsequently remove the logbook.Aircraft state entry.
    """

    initial = True

    dependencies = []

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.CreateModel(
                    name='Aircraft',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('model', models.CharField(max_length=100)),
                        ('manufacturer', models.CharField(blank=True, max_length=100)),
                    ],
                    options={
                        'verbose_name_plural': 'Aircraft',
                        'ordering': ['manufacturer', 'model'],
                        'db_table': 'logbook_aircraft',
                    },
                ),
            ],
        ),
    ]
