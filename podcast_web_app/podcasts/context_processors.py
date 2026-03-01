# podcasts/context_processors.py
from .models import Reply, SupportTicket  # assuming Reply is your proxy for Comment replies
from django.db.models import Q

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

def admin_ticket_counts(request):
    """
    Provides pending/in-progress support ticket counts for admins only.
    """
    pending_count = 0

    user = getattr(request, "user", None)
    if user and user.is_authenticated and (user.is_staff or user.is_superuser):
        pending_count = SupportTicket.objects.filter(
            status__in=["pending", "in_progress"]
        ).count()

    return {
        "admin_pending_tickets_count": pending_count,
    }
