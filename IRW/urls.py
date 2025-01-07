from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views
urlpatterns = [
    path('', views.index_view, name='index_view'),
    path('file_analysis/<str:recipe_id>/', views.file_analysis_view, name='file_analysis_view'),
    path('about/', views.about_view, name='about_view'),
    ]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)