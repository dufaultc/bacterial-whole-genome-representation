"""
Useful stuff

"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Return project root directory
    """
    return Path(__file__).parent.parent
