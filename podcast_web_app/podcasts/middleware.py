# podcasts/middleware.py
import logging
from datetime import datetime

class PageVisitMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.logger = logging.getLogger(__name__)

    def __call__(self, request):
        # Only log if the user is authenticated
        if request.user.is_authenticated:
            # Log the user's username, the path they visited, and the time
            self.logger.info(
                "User %s visited %s at %s",
                request.user.username,
                request.path,
                datetime.now().isoformat()
            )
        response = self.get_response(request)
        return response
