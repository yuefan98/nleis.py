__version__ = "0.3"

try:
    from .nleis import *  # noqa: F401, F403
except ImportError as e:
    # Print a warning
    print(f"Warning: Could not import nleis due to: {e}")
