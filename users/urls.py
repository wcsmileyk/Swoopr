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
    path('flight/upload/results/', views.upload_results_view, name='upload_results'),
    path('flight/<int:flight_id>/', views.flight_detail_view, name='flight_detail'),
    path('flight/<int:flight_id>/toggle-incorrect/', views.toggle_data_incorrect_view, name='toggle_data_incorrect'),
    path('flight/<int:flight_id>/delete/', views.delete_flight_view, name='delete_flight'),
    path('flights/bulk-delete/', views.bulk_delete_flights_view, name='bulk_delete_flights'),

    # Canopy management
    path('canopy/add/', views.add_canopy_view, name='add_canopy'),
    path('canopy/<int:canopy_id>/edit/', views.edit_canopy_view, name='edit_canopy'),
    path('canopy/<int:canopy_id>/delete/', views.delete_canopy_view, name='delete_canopy'),

    # Privacy and social features
    path('privacy/', views.privacy_settings_view, name='privacy_settings'),
    path('privacy/bulk-update/', views.bulk_privacy_update_view, name='bulk_privacy_update'),
    path('flight/<int:flight_id>/toggle-privacy/', views.toggle_flight_privacy_view, name='toggle_flight_privacy'),

    # Public features (no login required)
    path('search/', views.user_search_view, name='user_search'),
    path('profile/<str:username>/', views.public_profile_view, name='public_profile'),
    path('swoops/', views.public_swoops_view, name='public_swoops'),
]