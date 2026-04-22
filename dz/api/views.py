import datetime

from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from dz.models import DailyRoster, JumpRunConditions, Load, LoadSlot
from dz.services import load_service
from organizations.models import Dropzone

from .permissions import IsDZAdmin, IsDZStaff
from .serializers import (
    JumpRunConditionsCreateSerializer,
    JumpRunConditionsSerializer,
    LoadCreateSerializer,
    LoadSerializer,
    LoadSlotCreateSerializer,
    LoadSlotSerializer,
    LoadSlotUpdateSerializer,
    LoadUpdateSerializer,
    RosterEntrySerializer,
)


def _get_dz(dz_id):
    return get_object_or_404(Dropzone, pk=dz_id)


def _parse_date(request):
    date_str = request.query_params.get('date')
    if date_str:
        try:
            return datetime.date.fromisoformat(date_str)
        except (ValueError, TypeError):
            pass
    return timezone.localdate()


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------

class LoadListView(APIView):
    permission_classes = [IsDZStaff]

    def get(self, request, dz_id):
        dz = _get_dz(dz_id)
        date = _parse_date(request)
        loads = (
            Load.objects
            .filter(dropzone=dz, date=date)
            .select_related('aircraft', 'created_by')
            .prefetch_related('slots__user', 'slots__added_by')
            .order_by('load_number')
        )
        return Response(LoadSerializer(loads, many=True).data)

    def post(self, request, dz_id):
        dz = _get_dz(dz_id)
        serializer = LoadCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        load = load_service.create_load(
            dz=dz,
            date=serializer.validated_data['date'],
            created_by=request.user,
            aircraft=serializer.validated_data.get('aircraft'),
        )
        return Response(LoadSerializer(load).data, status=status.HTTP_201_CREATED)


class LoadDetailView(APIView):
    permission_classes = [IsDZStaff]

    def _get_load(self, dz_id, load_id):
        return get_object_or_404(Load, pk=load_id, dropzone_id=dz_id)

    def get(self, request, dz_id, load_id):
        load = self._get_load(dz_id, load_id)
        return Response(LoadSerializer(load).data)

    def patch(self, request, dz_id, load_id):
        load = self._get_load(dz_id, load_id)
        serializer = LoadUpdateSerializer(load, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(LoadSerializer(load).data)


# ---------------------------------------------------------------------------
# Load transitions — each is a dedicated endpoint so the action is explicit
# ---------------------------------------------------------------------------

class LoadCallView(APIView):
    permission_classes = [IsDZStaff]

    def post(self, request, dz_id, load_id):
        load = get_object_or_404(Load, pk=load_id, dropzone_id=dz_id)
        try:
            load_service.call_load(load, called_by=request.user)
        except ValueError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(LoadSerializer(load).data)


class LoadUncallView(APIView):
    permission_classes = [IsDZStaff]

    def post(self, request, dz_id, load_id):
        load = get_object_or_404(Load, pk=load_id, dropzone_id=dz_id)
        try:
            load_service.uncall_load(load, user=request.user)
        except ValueError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(LoadSerializer(load).data)


class LoadDepartView(APIView):
    permission_classes = [IsDZStaff]

    def post(self, request, dz_id, load_id):
        load = get_object_or_404(Load, pk=load_id, dropzone_id=dz_id)
        try:
            load_service.depart_load(load, user=request.user)
        except ValueError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(LoadSerializer(load).data)


class LoadLandView(APIView):
    permission_classes = [IsDZStaff]

    def post(self, request, dz_id, load_id):
        load = get_object_or_404(Load, pk=load_id, dropzone_id=dz_id)
        try:
            load_service.land_load(load, user=request.user)
        except ValueError as e:
            return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(LoadSerializer(load).data)


# ---------------------------------------------------------------------------
# Load slots
# ---------------------------------------------------------------------------

class LoadSlotListView(APIView):
    permission_classes = [IsDZStaff]

    def _get_load(self, dz_id, load_id):
        return get_object_or_404(Load, pk=load_id, dropzone_id=dz_id)

    def get(self, request, dz_id, load_id):
        load = self._get_load(dz_id, load_id)
        slots = load.slots.select_related('user', 'added_by').order_by('group_key', 'role')
        return Response(LoadSlotSerializer(slots, many=True).data)

    def post(self, request, dz_id, load_id):
        load = self._get_load(dz_id, load_id)
        if load.status not in ('building', 'on_call'):
            return Response(
                {'detail': 'Cannot add slots to a load that has departed or landed.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        serializer = LoadSlotCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        slot = serializer.save(load=load, added_by=request.user)
        return Response(LoadSlotSerializer(slot).data, status=status.HTTP_201_CREATED)


class LoadSlotDetailView(APIView):
    permission_classes = [IsDZStaff]

    def _get_slot(self, dz_id, load_id, slot_id):
        return get_object_or_404(LoadSlot, pk=slot_id, load_id=load_id, load__dropzone_id=dz_id)

    def patch(self, request, dz_id, load_id, slot_id):
        slot = self._get_slot(dz_id, load_id, slot_id)
        serializer = LoadSlotUpdateSerializer(slot, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(LoadSlotSerializer(slot).data)

    def delete(self, request, dz_id, load_id, slot_id):
        slot = self._get_slot(dz_id, load_id, slot_id)
        if slot.load.status not in ('building', 'on_call'):
            return Response(
                {'detail': 'Cannot remove slots from a load that has departed or landed.'},
                status=status.HTTP_400_BAD_REQUEST,
            )
        slot.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------------------
# Jump run conditions
# ---------------------------------------------------------------------------

class JumpRunView(APIView):
    permission_classes = [IsDZStaff]

    def get(self, request, dz_id):
        dz = _get_dz(dz_id)
        date = _parse_date(request)
        entry = (
            JumpRunConditions.objects
            .filter(dropzone=dz, date=date)
            .order_by('-set_at')
            .first()
        )
        if not entry:
            return Response(None)
        return Response(JumpRunConditionsSerializer(entry).data)

    def post(self, request, dz_id):
        """Log a new jump run condition entry (append-only)."""
        dz = _get_dz(dz_id)
        serializer = JumpRunConditionsCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        entry = serializer.save(dropzone=dz, set_by=request.user)
        return Response(JumpRunConditionsSerializer(entry).data, status=status.HTTP_201_CREATED)


# ---------------------------------------------------------------------------
# Roster (read-only — writes go through the session views)
# ---------------------------------------------------------------------------

class RosterView(APIView):
    permission_classes = [IsDZStaff]

    def get(self, request, dz_id):
        dz = _get_dz(dz_id)
        date = _parse_date(request)
        roster = (
            DailyRoster.objects
            .filter(dropzone=dz, date=date)
            .select_related('user')
            .prefetch_related('user__instructor_ratings')
            .order_by('user__last_name', 'user__first_name')
        )
        return Response(RosterEntrySerializer(roster, many=True).data)
