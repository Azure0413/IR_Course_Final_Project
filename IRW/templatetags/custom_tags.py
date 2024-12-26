from django import template

register = template.Library()

@register.filter
def index(sequence, position):
    try:
        return sequence[position]
    except (IndexError, TypeError):
        return None

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)