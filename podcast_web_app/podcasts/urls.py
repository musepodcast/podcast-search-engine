# podcasts/urls.py

from django.urls import path
from . import views
from .views import (
    SignUpView, ProfileView, ProfileUpdateView, CustomLoginView, 
    TwoFactorChallengeView, SecureDisable2FAView
)

from django.contrib.auth import views as auth_views
from django.views.generic.base import RedirectView
from django.contrib.staticfiles.storage import staticfiles_storage


app_name = 'podcasts'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),  # Home URL
    path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('podcasts/favicon.ico'))),
    path('favorites/', views.FavoritesListView.as_view(), name='favorites'),
    path('episode/<int:episode_id>/toggle_bookmark/', views.toggle_episode_bookmark, name='toggle_episode_bookmark'),
    path('episode/<int:episode_id>/update_episode_rating/', views.update_episode_rating, name='update_episode_rating'),
    path('bookmarks/', views.BookmarksListView.as_view(), name='bookmarks'),
    path('notifications/', views.NotificationsListView.as_view(), name='notifications'),
    path('channels/', views.ChannelListView.as_view(), name='channel_list'),  # Channel list URL
    path('episodes/', views.EpisodeListView.as_view(), name='episode_list'),  # Episode list URL
    path('search/', views.SearchResultsView.as_view(), name='search_results'),  # Search URL
    path('signup/', SignUpView.as_view(), name='signup'),
    path("login/", CustomLoginView.as_view(template_name='registration/login.html'), name="login"),
    path("login/2fa/", TwoFactorChallengeView.as_view(template_name='registration/login_2fa.html'), name="two_factor_challenge"),
    path('replies/', views.RepliesListView.as_view(), name='replies'),
    path('search_users/', views.search_users, name='search_users'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('profile_edit/', ProfileUpdateView.as_view(), name='profile_edit'),
    path('disable_2fa/', SecureDisable2FAView.as_view(), name='disable_2fa'),
    path('channel/<int:channel_id>/toggle_follow/', views.toggle_follow, name='toggle_follow'),
    path('channel/<int:channel_id>/toggle_notifications/', views.toggle_notifications, name='toggle_notifications'),
    path('channel/<int:channel_id>/update_rating/', views.update_rating, name='update_rating'),
    # Episode detail URL with sanitized title.
    path('<str:sanitized_channel_title>/<str:sanitized_episode_title>/', views.EpisodeDetailView.as_view(), name='episode_detail'),
    path('<str:sanitized_channel_title>/', views.ChannelDetailView.as_view(), name='channel_detail'),  # Channel detail URL
    # === New URL patterns for comments ===
    path('episode/<int:episode_id>/post_comment/', views.post_comment, name='post_comment'),
    path('episode/<int:episode_id>/comments/', views.get_comments, name='get_comments'),
    path('comment/<int:comment_id>/reaction/', views.comment_reaction, name='comment_reaction'),
    path('search_users/', views.search_users, name='search_users'),
]
# path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
# 