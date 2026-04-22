from django.urls import path

from . import views, manifest_views

urlpatterns = [
    path('', views.dz_redirect, name='dz_redirect'),
    path('<int:dz_id>/', views.dz_dashboard, name='dz_dashboard'),
    path('<int:dz_id>/staff/', views.dz_staff_list, name='dz_staff_list'),
    path('<int:dz_id>/staff/add/', views.dz_add_member, name='dz_add_member'),
    path('<int:dz_id>/staff/<int:user_id>/', views.dz_staff_detail, name='dz_staff_detail'),
    path('<int:dz_id>/staff/<int:user_id>/authorize/', views.dz_set_authorization, name='dz_set_authorization'),
    path('<int:dz_id>/staff/<int:user_id>/rating/', views.dz_set_rating, name='dz_set_rating'),
    path('<int:dz_id>/staff/<int:user_id>/membership/', views.dz_update_membership, name='dz_update_membership'),
    # Daily ops
    path('<int:dz_id>/roster/', views.dz_roster, name='dz_roster'),
    path('<int:dz_id>/checkin/', views.dz_checkin, name='dz_checkin'),
    path('<int:dz_id>/schedule/', views.dz_schedule, name='dz_schedule'),
    # Reservations
    path('<int:dz_id>/reservations/', views.dz_reservations, name='dz_reservations'),
    path('<int:dz_id>/reservations/new/', views.dz_reservation_new, name='dz_reservation_new'),
    path('<int:dz_id>/reservations/<int:reservation_id>/', views.dz_reservation_detail, name='dz_reservation_detail'),
    # Manifest board
    path('<int:dz_id>/manifest/', manifest_views.dz_manifest, name='dz_manifest'),
    path('<int:dz_id>/manifest/board/', manifest_views.dz_manifest_board, name='dz_manifest_board'),
    path('<int:dz_id>/manifest/loads/new/', manifest_views.dz_manifest_load_new, name='dz_manifest_load_new'),
    path('<int:dz_id>/manifest/loads/<int:load_id>/call/', manifest_views.dz_manifest_load_call, name='dz_manifest_load_call'),
    path('<int:dz_id>/manifest/loads/<int:load_id>/uncall/', manifest_views.dz_manifest_load_uncall, name='dz_manifest_load_uncall'),
    path('<int:dz_id>/manifest/loads/<int:load_id>/depart/', manifest_views.dz_manifest_load_depart, name='dz_manifest_load_depart'),
    path('<int:dz_id>/manifest/loads/<int:load_id>/land/', manifest_views.dz_manifest_load_land, name='dz_manifest_load_land'),
    path('<int:dz_id>/manifest/loads/<int:load_id>/slots/add/', manifest_views.dz_manifest_slot_add, name='dz_manifest_slot_add'),
    path('<int:dz_id>/manifest/loads/<int:load_id>/slots/<int:slot_id>/remove/', manifest_views.dz_manifest_slot_remove, name='dz_manifest_slot_remove'),
    path('<int:dz_id>/manifest/jump-run/', manifest_views.dz_manifest_jump_run, name='dz_manifest_jump_run'),
    # Students
    path('<int:dz_id>/students/', views.dz_student_list, name='dz_student_list'),
    path('<int:dz_id>/students/enroll/', views.dz_enroll_student, name='dz_enroll_student'),
    path('<int:dz_id>/students/<int:enrollment_id>/', views.dz_student_detail, name='dz_student_detail'),
    path('<int:dz_id>/students/<int:enrollment_id>/log/', views.dz_log_student_jump, name='dz_log_student_jump'),
]
