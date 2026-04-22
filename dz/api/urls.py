from django.urls import path

from . import views

urlpatterns = [
    # Loads
    path('<int:dz_id>/loads/', views.LoadListView.as_view(), name='api_load_list'),
    path('<int:dz_id>/loads/<int:load_id>/', views.LoadDetailView.as_view(), name='api_load_detail'),
    path('<int:dz_id>/loads/<int:load_id>/call/', views.LoadCallView.as_view(), name='api_load_call'),
    path('<int:dz_id>/loads/<int:load_id>/uncall/', views.LoadUncallView.as_view(), name='api_load_uncall'),
    path('<int:dz_id>/loads/<int:load_id>/depart/', views.LoadDepartView.as_view(), name='api_load_depart'),
    path('<int:dz_id>/loads/<int:load_id>/land/', views.LoadLandView.as_view(), name='api_load_land'),
    # Slots
    path('<int:dz_id>/loads/<int:load_id>/slots/', views.LoadSlotListView.as_view(), name='api_slot_list'),
    path('<int:dz_id>/loads/<int:load_id>/slots/<int:slot_id>/', views.LoadSlotDetailView.as_view(), name='api_slot_detail'),
    # Jump run
    path('<int:dz_id>/jump-run/', views.JumpRunView.as_view(), name='api_jump_run'),
    # Roster
    path('<int:dz_id>/roster/', views.RosterView.as_view(), name='api_roster'),
]
