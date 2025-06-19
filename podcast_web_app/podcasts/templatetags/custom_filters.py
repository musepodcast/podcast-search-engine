import re
import bleach
from django import template
from django.utils.safestring import mark_safe
from django.contrib.auth import get_user_model
from django.utils.html import strip_tags

register = template.Library()
User = get_user_model()

@register.filter
def flatten_html(value):
    """
    Remove all <p> (and any other unwanted block tags), collapse whitespace.
    """
    # kill <p> and </p> completely
    no_ps = re.sub(r'</?p[^>]*>', ' ', value)
    # collapse runs of whitespace
    collapsed = re.sub(r'\s+', ' ', no_ps).strip()
    return mark_safe(collapsed)


@register.filter
def clean_html(value):
    """
    Strip out anything but a few safe inline tags (<a>, <b>, <i>, <em>, <strong>, <ul>, <li>, <br>).
    """
    if not value:
        return ''
    allowed_tags = ['a','b','i','u','em','strong','br','ul','ol','li']
    allowed_attrs = {'a': ['href','title','target','rel']}
    cleaned = bleach.clean(
        value,
        tags=allowed_tags,
        attributes=allowed_attrs,
        strip=True,
        strip_comments=True
    )
    return mark_safe(cleaned)

@register.filter
def highlight_sentence(text, search_term):
    """
    Wrap each match of `search_term` in a <span class="highlight">…</span>.
    """
    if not search_term:
        return mark_safe(text)
    terms = [re.escape(t) for t in search_term.split()]
    pattern = re.compile(r'\b(' + '|'.join(terms) + r')\b', re.IGNORECASE)
    highlighted = pattern.sub(
        lambda m: f'<span class="highlight">{m.group(0)}</span>',
        text
    )
    return mark_safe(highlighted)

@register.filter(name='contains')
def contains(text, search_term):
    """
    Returns True if the search_term is found within the text (case-insensitive), else False.
    """
    if not search_term:
        return False
    return search_term.lower() in text.lower()

@register.filter(name='get_list')
def get_list(querydict, key):
    """
    Returns a list of values for the given key from a QueryDict.
    This allows you to use the 'getlist' method in templates.
    """
    return querydict.getlist(key)

@register.filter
def number_format(value):
    """Formats a number into a human-friendly format using k, M, or B."""
    try:
        value = int(value)
    except (ValueError, TypeError):
        return value
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.0f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.0f}M"
    elif value >= 1_000:
        return f"{value/1_000:.0f}k"
    else:
        return str(value)
    
@register.filter
def highlight_mentions(text):
    """
    Searches for @username patterns in the text and, if the username exists,
    wraps the mention in a span with class "mention-highlight".
    """
    def repl(match):
        username = match.group(1)
        try:
            # Validate that the username exists.
            User.objects.get(username=username)
            return f'<span class="mention-highlight">@{username}</span>'
        except User.DoesNotExist:
            # If no such user exists, keep it unchanged.
            return match.group(0)
    highlighted = re.sub(r'@(\w+)', repl, text)
    return mark_safe(highlighted)