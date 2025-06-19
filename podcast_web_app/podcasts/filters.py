# podcasts/filters.py

import django_filters
from .models import Episode

class EpisodeFilter(django_filters.FilterSet):
    publication_date = django_filters.DateFromToRangeFilter()
    duration = django_filters.CharFilter(lookup_expr='icontains')
    explicit = django_filters.BooleanFilter()

    class Meta:
        model = Episode
        fields = ['publication_date', 'duration', 'explicit']
