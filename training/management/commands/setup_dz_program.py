"""
Create a default ProgramJump structure for a dropzone, using the USPA-recommended
jump structure stored in each USPACategory.recommended_jumps.

Each category can produce multiple ProgramJump rows (one per jump per method_group),
e.g. Cat A produces 1 AFF_Tandem jump + 2 IAD_SL jumps.

Run seed_program_levels first so USPACategory rows exist.

Usage:
    python manage.py setup_dz_program <dz_id> --program AFF
    python manage.py setup_dz_program <dz_id> --program AFF --overwrite
"""
from django.core.management.base import BaseCommand, CommandError

from organizations.models import Dropzone
from training.models import ProgramJump, USPACategory


class Command(BaseCommand):
    help = 'Create default ProgramJump rows for a dropzone from USPACategory.recommended_jumps'

    def add_arguments(self, parser):
        parser.add_argument('dz_id', type=int, help='Dropzone primary key')
        parser.add_argument('--program', default='AFF', help='Program type (default: AFF)')
        parser.add_argument('--overwrite', action='store_true', help='Overwrite existing jumps')

    def handle(self, *args, **options):
        try:
            dz = Dropzone.objects.get(pk=options['dz_id'])
        except Dropzone.DoesNotExist:
            raise CommandError(f"No dropzone with id {options['dz_id']}")

        program_type = options['program']
        overwrite = options['overwrite']

        categories = USPACategory.objects.filter(program_type=program_type).order_by('order')
        if not categories.exists():
            raise CommandError(
                f"No USPACategory rows found for program '{program_type}'. "
                f"Run 'seed_program_levels' first."
            )

        created = updated = skipped = 0

        for cat in categories:
            recommended = cat.recommended_jumps  # dict: {method_group: [jump, ...]}

            if not recommended:
                self.stdout.write(f"  {cat.code}: no recommended jumps — skipping")
                continue

            for method_group, jumps in recommended.items():
                if not jumps:
                    continue

                for jump_def in jumps:
                    jump_number = jump_def['jump_number']
                    defaults = {
                        'name': jump_def.get('name', ''),
                        'allowed_methods': jump_def.get('allowed_methods', []),
                        'dive_flow': jump_def.get('dive_flow', []),
                        'assigned_criteria': cat.criteria,
                    }

                    obj, was_created = ProgramJump.objects.get_or_create(
                        dropzone=dz,
                        category=cat,
                        method_group=method_group,
                        jump_number=jump_number,
                        defaults=defaults,
                    )

                    if was_created:
                        created += 1
                    elif overwrite:
                        for field, value in defaults.items():
                            setattr(obj, field, value)
                        obj.save()
                        updated += 1
                    else:
                        skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f'{dz.name} — {program_type}: created {created}, updated {updated}, skipped {skipped} jumps.'
        ))
        if skipped:
            self.stdout.write('Use --overwrite to update existing jumps.')
