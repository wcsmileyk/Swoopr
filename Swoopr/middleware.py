"""
Custom middleware for enhanced error logging and monitoring
"""

import logging
import traceback
import json
from django.http import JsonResponse
from django.conf import settings
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger('swoopr')


class ErrorLoggingMiddleware(MiddlewareMixin):
    """
    Middleware to log detailed information about 500 errors
    while keeping sensitive information secure
    """

    def process_exception(self, request, exception):
        """
        Log detailed information about exceptions that cause 500 errors
        """
        # Get request metadata (safe for logging)
        request_data = {
            'method': request.method,
            'path': request.path,
            'user': str(request.user) if hasattr(request, 'user') and request.user.is_authenticated else 'Anonymous',
            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
            'remote_addr': self._get_client_ip(request),
            'referer': request.META.get('HTTP_REFERER', ''),
            'content_type': request.content_type,
        }

        # Add query parameters (filter out sensitive data)
        if request.GET:
            safe_params = {}
            for key, value in request.GET.items():
                # Skip potentially sensitive parameters
                if key.lower() not in ['password', 'token', 'key', 'secret']:
                    safe_params[key] = value
            request_data['query_params'] = safe_params

        # Add POST data size (not content for security)
        if request.method == 'POST':
            request_data['post_data_size'] = len(request.body) if hasattr(request, 'body') else 0
            request_data['files_uploaded'] = len(request.FILES) if hasattr(request, 'FILES') else 0

        # Format the error message
        error_msg = (
            f"500 Error: {exception.__class__.__name__}: {str(exception)}\n"
            f"Request: {json.dumps(request_data, indent=2)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

        # Log the error
        logger.error(error_msg, extra={
            'request': request,
            'exception_type': exception.__class__.__name__,
            'exception_message': str(exception),
            'request_path': request.path,
            'request_method': request.method,
            'user_id': request.user.id if hasattr(request, 'user') and request.user.is_authenticated else None,
        })

        # Don't return a response - let Django handle it normally
        return None

    def _get_client_ip(self, request):
        """Get the client IP address from the request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


class RequestLoggingMiddleware(MiddlewareMixin):
    """
    Optional middleware to log all requests with non-standard status codes
    (Enable this if you want even more verbose logging)
    """

    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)

    def __call__(self, request):
        response = self.get_response(request)

        # Log requests with problematic status codes (but not 404/403 as requested)
        if response.status_code >= 500:
            logger.warning(
                f"HTTP {response.status_code}: {request.method} {request.path} "
                f"User: {request.user if hasattr(request, 'user') else 'Unknown'} "
                f"IP: {self._get_client_ip(request)}"
            )
        elif response.status_code in [400, 401, 405, 408, 409, 410, 413, 422, 429]:
            logger.info(
                f"HTTP {response.status_code}: {request.method} {request.path} "
                f"User: {request.user if hasattr(request, 'user') else 'Unknown'}"
            )

        return response

    def _get_client_ip(self, request):
        """Get the client IP address from the request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip