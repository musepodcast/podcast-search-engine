# podcasts/context_processors.py
from .models import Reply  # assuming Reply is your proxy for Comment replies

def unseen_replies(request):
    if request.user.is_authenticated:
        username = request.user.username
        # Count only those replies that mention the user and where the user is NOT in seen_by.
        count = Reply.objects.filter(
                    text__icontains='@' + username
                ).exclude(
                    seen_by=request.user
                ).count()
        return {'unseen_replies_count': count}
    return {}
