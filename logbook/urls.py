from django.urls import path
from . import views

urlpatterns = [
    path('', views.jump_list, name='jump_list'),
    path('log/', views.log_jumps, name='log_jumps'),
    path('upload/', views.upload_wizard_step1, name='logbook_upload'),
    path('upload/log/', views.upload_wizard_step2, name='logbook_upload_log'),
    path('import/', views.import_csv_step1, name='logbook_import'),
    path('import/map/', views.import_csv_step2, name='logbook_import_map'),
    path('import/confirm/', views.import_csv_confirm, name='logbook_import_confirm'),
    path('<int:pk>/edit/', views.edit_jump, name='edit_jump'),
    path('<int:pk>/delete/', views.delete_jump, name='delete_jump'),
    path('<int:pk>/attach-gps/', views.attach_gps, name='attach_gps'),
    path('<int:pk>/link-flight/', views.link_flight, name='link_flight'),
    path('<int:pk>/unlink-flight/', views.unlink_flight, name='unlink_flight'),
]
