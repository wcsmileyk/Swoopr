from django.urls import path

from . import views

app_name = 'courses'

urlpatterns = [
    path('', views.course_list_view, name='list'),
    path('import/', views.course_import_upload_view, name='import_upload'),
    path('import/<uuid:import_id>/', views.course_import_preview_view, name='import_preview'),
    path('<uuid:course_set_id>/', views.course_detail_view, name='detail'),
    path('<uuid:course_set_id>/visibility/', views.course_visibility_view, name='visibility'),
    path('<uuid:course_set_id>/archive/', views.course_archive_view, name='archive'),
    path('flights/<int:flight_id>/apply/', views.apply_course_to_flight_view, name='apply_to_flight'),
    path('flights/<int:flight_id>/analysis/<int:analysis_id>/', views.flight_analysis_view, name='flight_analysis'),
]
