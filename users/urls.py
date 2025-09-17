from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),

    # User dashboard and profile
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('profile/', views.profile_view, name='profile'),
    path('flights/', views.flights_view, name='flights'),
    path('flight/upload/', views.upload_flight_view, name='upload_flight'),
    path('flight/<int:flight_id>/', views.flight_detail_view, name='flight_detail'),
    path('flight/<int:flight_id>/toggle-incorrect/', views.toggle_data_incorrect_view, name='toggle_data_incorrect'),
    path('flight/<int:flight_id>/delete/', views.delete_flight_view, name='delete_flight'),
    path('flights/bulk-delete/', views.bulk_delete_flights_view, name='bulk_delete_flights'),

    # Canopy management
    path('canopy/add/', views.add_canopy_view, name='add_canopy'),
    path('canopy/<int:canopy_id>/edit/', views.edit_canopy_view, name='edit_canopy'),
    path('canopy/<int:canopy_id>/delete/', views.delete_canopy_view, name='delete_canopy'),
]