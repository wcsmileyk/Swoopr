from django.urls import path
from . import views

app_name = 'flights'

urlpatterns = [
    # Competition gate views
    path('gates/<int:gate_id>/map/', views.gate_map_view, name='gate_map'),
    path('gates/<int:gate_id>/data/', views.gate_course_data, name='gate_course_data'),
]
