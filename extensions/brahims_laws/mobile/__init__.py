"""
Brahim Onion Agent - Mobile SDK

Cross-platform mobile deployment for Android (APK) and iOS.
Supports: Kivy, BeeWare, Chaquopy, and REST API modes.

Author: Elias Oulad Brahim
DOI: 10.5281/zenodo.18356196
"""

from .api_server import BrahimAPIServer, create_app
from .config import MobileConfig, APKConfig

__all__ = [
    "BrahimAPIServer",
    "create_app",
    "MobileConfig",
    "APKConfig",
]
