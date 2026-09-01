"""
app/cli - Visual command-line interface, Rich UI, and demo recording mode.
"""

from app.cli.app import cli_app
from app.cli.demo_recorder import DemoRecorder
from app.cli.ui import ConsoleUI

__all__ = ["cli_app", "ConsoleUI", "DemoRecorder"]
