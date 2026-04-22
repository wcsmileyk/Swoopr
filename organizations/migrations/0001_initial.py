from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Registers organizations.Dropzone in the migration state.
    The logbook_dropzone table already exists — no database operations needed.
    logbook/0005 will subsequently remove the logbook.Dropzone state entry.
    """

    initial = True

    dependencies = []

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
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
                    options={
                        'ordering': ['name'],
                        'db_table': 'logbook_dropzone',
                    },
                ),
            ],
        ),
    ]
