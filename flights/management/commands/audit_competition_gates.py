"""
Read-only audit of legacy CompetitionGate rows ahead of the course-system rebuild.

Reports, for every row: source file availability, owner, parse status,
coordinate plausibility, and how many flights reference it. Makes no changes.

Usage:
    python manage.py audit_competition_gates
"""

from django.core.management.base import BaseCommand
from flights.models import CompetitionGate


def _plausible_latlon(lat, lon):
    if lat is None or lon is None:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


class Command(BaseCommand):
    help = 'Read-only audit of legacy CompetitionGate rows (source, owner, parse status, coordinates, usage)'

    def handle(self, *args, **options):
        gates = CompetitionGate.objects.select_related('created_by').order_by('id')
        total = gates.count()

        if total == 0:
            self.stdout.write(self.style.WARNING('No CompetitionGate rows found'))
            return

        ownerless = 0
        missing_file = 0
        unparsed = 0
        implausible_coords = 0
        assigned_flight_count = 0

        self.stdout.write(f'Auditing {total} CompetitionGate rows...\n')

        for gate in gates:
            issues = []

            owner = gate.created_by.username if gate.created_by else None
            if owner is None:
                ownerless += 1
                issues.append('NO OWNER')

            has_file = bool(gate.gate_file) and self._file_exists(gate)
            if not has_file:
                missing_file += 1
                issues.append('SOURCE FILE MISSING')

            if not gate.is_parsed:
                unparsed += 1
                issues.append('NOT PARSED')
            elif gate.parse_error:
                issues.append(f'PARSE ERROR RECORDED: {gate.parse_error[:80]}')

            if not _plausible_latlon(gate.center_lat, gate.center_lon):
                implausible_coords += 1
                issues.append('IMPLAUSIBLE/MISSING CENTER COORDINATES')

            flight_count = gate.flights.count()
            assigned_flight_count += flight_count

            status = self.style.SUCCESS('OK') if not issues else self.style.WARNING('REVIEW')
            self.stdout.write(
                f'[{status}] id={gate.id} name={gate.name!r} type={gate.gate_type} '
                f'owner={owner or "-"} flights={flight_count}'
            )
            for issue in issues:
                self.stdout.write(f'    - {issue}')

        self.stdout.write('\nSummary:')
        self.stdout.write(f'  Total rows:               {total}')
        self.stdout.write(f'  Ownerless:                {ownerless}')
        self.stdout.write(f'  Missing source file:      {missing_file}')
        self.stdout.write(f'  Not parsed:               {unparsed}')
        self.stdout.write(f'  Implausible coordinates:  {implausible_coords}')
        self.stdout.write(f'  Flights referencing rows: {assigned_flight_count}')
        self.stdout.write(
            self.style.WARNING(
                '\nThis command makes no changes. Ownerless or flagged rows should go to '
                'admin quarantine during migration, not be trusted or deleted automatically.'
            )
        )

    @staticmethod
    def _file_exists(gate):
        try:
            return gate.gate_file.storage.exists(gate.gate_file.name)
        except Exception:
            return False
