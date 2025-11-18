"""Logging configuration utilities for the general_backend package.

This module provides optional logging configuration utilities for developers
who want to configure logging for the general_backend package and its
dependencies. For typical end-users, no action is needed as the package will
use the root logger's configuration by default.

Functions
---------
get_logger : function
    Get a package-scoped logger.
    Usage: logger = get_logger(__name__)

set_logger_level_for_dependency : function
    Set logging level for a specific dependency package.
    Usage: set_logger_level_for_dependency(
        'some_dependency', logging.ERROR
    )

configure_default_logging : function
    Configure default logging for the package and root logger.
    Usage: configure_default_logging(
        pckg_level=logging.INFO, root_level=logging.WARNING
    )

Notes
-----
Contents adapted from standard logging practices.
The module ensures proper logger hierarchy and prevents duplicate log
messages by managing handler propagation.

Authors
Johannes Fjeldså
"""

import logging

from general_backend import PACKAGE_LOGGER_NAME


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a package-scoped logger.
    Example: get_logger('module') -> 'reversclim.module'

    Parameters
    ----------
    name : str | None, optional
        name for logger object.

    Returns
    -------
    logging.Logger : logging.Logger
        Logger object with the name.
    """
    name = name or ""
    if name.startswith(PACKAGE_LOGGER_NAME):
        logger_name = name
    else:
        logger_name = PACKAGE_LOGGER_NAME + f".{name}"

    return logging.getLogger(logger_name)


def set_logger_level_for_dependency(dependency_name: str, level: int) -> None:
    """Set the logging level for a specific dependency package.

    Parameters
    ----------
    dependency_name : str
        Name of the dependency package (e.g., 'some_dependency').
    level : int
        Logging level to set for the dependency package (e.g., logging.ERROR).
    """
    logger = logging.getLogger(dependency_name)
    logger.setLevel(level)


LOG_CONFIG_MSG = """
Configuring default logging for general_backend package:
* Package logger level: {pckg_level}
* Root logger level: {root_level}
To change the logging level for a specific dependency, use
set_logger_level_for_dependency importable from
general_backend.utils.general.setup_logging.py.
"""


def configure_default_logging(
    pckg_level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    root_level: int = logging.WARNING,
    supress_log_config_msg: bool = False,
) -> None:
    """Configure default logging for the package and root logger.

    Parameters
    ----------
    pckg_level : int, optional
        Logging level for the general_backend package logger,
        by default logging.INFO
    fmt : str, optional
        Logging format string,
        by default '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    root_level : int, optional
        Logging level for the root logger, by default logging.WARNING
    supress_log_config_msg: bool, optional
        Whether to suppress the logging configuration message, by default False
    """
    # Configure the root logger centrally. Using basicConfig is the
    # simplest way to ensure the root handler/level are set consistently.
    logging.basicConfig(level=root_level, format=fmt)

    # Ensure the root logger's level is explicitly set (basicConfig might
    # not change it if handlers already existed in some environments).
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)

    # Configure the package logger
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    pkg_logger.setLevel(pckg_level)

    # Remove existing StreamHandlers on the package logger to avoid duplicates,
    # then add a single StreamHandler with the requested formatter.
    for h in list(pkg_logger.handlers):
        if isinstance(h, logging.StreamHandler):
            pkg_logger.removeHandler(h)

    handler = logging.StreamHandler()
    fmt_to_use = (
        fmt
        if fmt is not None
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt_to_use))
    handler.setLevel(pckg_level)
    pkg_logger.addHandler(handler)

    # Prevent package logs from propagating up to the root handler and being
    # printed twice (package handler + root handler).
    pkg_logger.propagate = False

    if not supress_log_config_msg:
        print(
            LOG_CONFIG_MSG.format(pckg_level=pckg_level, root_level=root_level)
        )
