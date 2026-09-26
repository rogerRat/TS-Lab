"""Presentation-only V7 aliases; persisted arm identities stay unchanged."""

DISPLAY_NAMES = {'grad-cos': 'gradcos-legacy'}


def display_arm(arm):
    return DISPLAY_NAMES.get(arm, arm)
