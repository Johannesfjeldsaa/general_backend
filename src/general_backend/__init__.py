# Package-level logger setup:
# we just provide a logger and a NullHandler
import logging
from pathlib import Path

# Use a constant package name so logs are consistently tagged
PACKAGE_LOGGER_NAME = "general_backend"
logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# Defensive logging setup - add a NullHandler to:
# * avoid "No handler found" warnings if the application using the package
#   has not configured logging.
# * allow the package logger to propagate messages to the root logger
#   if the application has configured logging.
logger.addHandler(logging.NullHandler())

# Export the package logger for convenience
__all__ = ["logger", "PACKAGE_LOGGER_NAME"]

# get src and project root paths
_current_file_path = Path(__file__).resolve()
SRC_PATH = _current_file_path.parent.parent  # src
PROJECT_ROOT_PATH = SRC_PATH.parent  # project root
