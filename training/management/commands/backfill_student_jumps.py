"""
Backfill training.StudentJump rows from the legacy StudentSignoff records
(and report on PendingInstructorRequest rows) so the newer eligibility/
progression UI has data to work with for students trained before StudentJump
existed.

This is a one-time, human-reviewed migration — NOT run automatically. Always
review the dry-run report before passing --apply, and expect some rows to be
skipped (logged as such) rather than guessed at.

What gets backfilled:
- StudentSignoff rows whose jump has no linked StudentJump yet. Each becomes
  a StudentJump with outcome='pass', instructor=signoff.signed_by (a real FK
  already, not just the snapshotted instructor_name string), and criteria
  copied from signoff.criteria_passed. A StudentEnrollment is get_or_create'd
  from jump.user/jump.dropzone/jump.student_jump_method. A ProgramJump is
  matched by (dropzone, category=matching USPACategory, is_active) — since
  the legacy schema never recorded which specific ProgramJump number was
  attempted within a category, the EARLIEST active ProgramJump in that
  category/method group is used as a best-effort match and every such row is
  flagged in the report. Rows where no matching ProgramJump exists at all
  (DZ never ran setup_dz_program) are skipped, not guessed.

What is NOT backfilled:
- PendingInstructorRequest rows (unsigned, in-flight requests). These are
  only counted in the report. They represent submitted-but-not-yet-approved
  jumps with no settled outcome/criteria — creating a StudentJump for them
  would fabricate a sign-off that never happened. Let them resolve naturally
  through the existing pending-request flow, or handle manually.

Usage:
    python manage.py backfill_student_jumps            # dry run, report only
    python manage.py backfill_student_jumps --apply     # actually create rows
"""

from django.core.management.base import BaseCommand
from django.db import transaction

from training.models import (
    PendingInstructorRequest, ProgramJump, StudentEnrollment, StudentJump,
    StudentSignoff, USPACategory,
)

# logbook.JUMP_METHOD_CHOICES ('AFF', 'Tandem', 'IAD_SL', 'Coach') never
# distinguished 2-instructor vs solo JM, or IAD vs static line, the way
# training.STUDENT_JUMP_METHOD_CHOICES does. These are the closest/most
# common defaults for each — genuinely approximate, flagged in row notes.
LEGACY_METHOD_TO_STUDENT_JUMP_METHOD = {
    'AFF': 'AFF_2I',
    'Tandem': 'Tandem',
    'IAD_SL': 'IAD',
    'Coach': 'Coach',
}
APPROXIMATE_METHODS = {'AFF', 'IAD_SL'}


class Command(BaseCommand):
    help = 'Backfill StudentJump rows from legacy StudentSignoff records (dry-run by default).'

    def add_arguments(self, parser):
        parser.add_argument(
            '--apply', action='store_true',
            help='Actually create StudentJump rows. Without this flag, only reports what would happen.',
        )

    def handle(self, *args, **options):
        apply = options['apply']

        signoffs = (
            StudentSignoff.objects
            .filter(jump__student_jump__isnull=True)
            .select_related('jump', 'jump__user', 'jump__dropzone', 'signed_by')
            .order_by('signed_at')
        )

        created, skipped_no_category, skipped_no_program_jump, approximate_matches = 0, 0, 0, 0

        for signoff in signoffs:
            jump = signoff.jump
            if not jump.student_jump_method or not jump.student_category:
                skipped_no_category += 1
                self.stdout.write(self.style.WARNING(
                    f'SKIP signoff {signoff.pk} (jump #{jump.jump_number}): '
                    f'jump has no student_jump_method/student_category to map from.'
                ))
                continue

            category = USPACategory.objects.filter(
                program_type=jump.student_jump_method, code=jump.student_category,
            ).first()
            if not category:
                skipped_no_category += 1
                self.stdout.write(self.style.WARNING(
                    f'SKIP signoff {signoff.pk} (jump #{jump.jump_number}): '
                    f'no USPACategory for program_type={jump.student_jump_method} code={jump.student_category}.'
                ))
                continue

            program_jump = (
                ProgramJump.objects
                .filter(dropzone=jump.dropzone, category=category, is_active=True)
                .order_by('method_group', 'jump_number')
                .first()
            )
            if not program_jump:
                skipped_no_program_jump += 1
                self.stdout.write(self.style.WARNING(
                    f'SKIP signoff {signoff.pk} (jump #{jump.jump_number}): '
                    f'no active ProgramJump at {jump.dropzone} for category {category}. '
                    f'Has this DZ run setup_dz_program?'
                ))
                continue

            approximate_matches += 1
            self.stdout.write(
                f'{"APPLY" if apply else "DRY-RUN"}: signoff {signoff.pk} (jump #{jump.jump_number}, '
                f'{jump.user.username}) -> ProgramJump {program_jump} '
                f'[best-effort match — legacy data has no jump-level granularity within the category]'
            )

            if apply:
                jump_method = LEGACY_METHOD_TO_STUDENT_JUMP_METHOD[jump.student_jump_method]
                notes = f'Backfilled from StudentSignoff {signoff.pk}.'
                if jump.student_jump_method in APPROXIMATE_METHODS:
                    notes += (
                        f' jump_method approximated as {jump_method} — legacy data only '
                        f'recorded "{jump.student_jump_method}", not the specific sub-method.'
                    )
                with transaction.atomic():
                    enrollment, _ = StudentEnrollment.objects.get_or_create(
                        student=jump.user, dropzone=jump.dropzone, program_type=jump.student_jump_method,
                        defaults={'status': 'active'},
                    )
                    StudentJump.objects.create(
                        enrollment=enrollment,
                        jump=jump,
                        program_jump=program_jump,
                        instructor=signoff.signed_by,
                        jump_method=jump_method,
                        jump_date=jump.date,
                        outcome='pass',
                        criteria_passed=signoff.criteria_passed,
                        signed_off_by=signoff.signed_by,
                        signed_off_at=signoff.signed_at,
                        notes=notes,
                    )
                created += 1

        pending_count = PendingInstructorRequest.objects.count()

        self.stdout.write('')
        count = created if apply else approximate_matches
        self.stdout.write(self.style.SUCCESS(
            f'{"Created" if apply else "Would create"}: {count} StudentJump row(s) '
            f'(all approximate matches — see notes above).'
        ))
        self.stdout.write(f'Skipped (no matching USPACategory): {skipped_no_category}')
        self.stdout.write(f'Skipped (no matching ProgramJump): {skipped_no_program_jump}')
        self.stdout.write(f'PendingInstructorRequest rows NOT backfilled (unsigned, left as-is): {pending_count}')
        if not apply:
            self.stdout.write(self.style.WARNING('\nDry run only — re-run with --apply to create these rows.'))
