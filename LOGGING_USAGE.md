# Logging Usage Guide

This document explains how to use the `general_backend` package's logging system in different scenarios.

## As a Dependency Package (Recommended Default)

When using `general_backend` as a dependency in your main project, **no additional configuration is needed**. The package will respect your main project's logging configuration.

### In your main project:

```python
import logging

# Configure your main project's logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import and use general_backend - it will automatically use your logging config
from general_backend.masking import mask_utils
from general_backend.logging.setup_logging import get_logger

# Your project's logger
logger = get_logger('my_project.module')

# The package will use your logging configuration
# Package logs will propagate to your root logger
```

### Controlling package log levels:

```python
from general_backend.logging.setup_logging import configure_package_logger

# Set the package to only show warnings and errors
configure_package_logger(level=logging.WARNING)

# Or completely silence the package (still propagates to your root logger)
configure_package_logger(level=logging.CRITICAL)
```

## As a Standalone Package

When using `general_backend` as the main application (not as a dependency):

```python
from general_backend.logging.setup_logging import configure_standalone_logging

# Configure full logging for standalone use
configure_standalone_logging(
    pckg_level=logging.INFO,     # Package logger level
    root_level=logging.WARNING,  # Root logger level
    suppress_log_config_msg=False  # Show configuration message
)

# Now use the package normally
from general_backend.masking import mask_utils
```

## Advanced Configuration

### Dependency-friendly package configuration:

```python
from general_backend.logging.setup_logging import configure_package_logger

# Configure package logger to work with your main project
configure_package_logger(
    level=logging.INFO,
    propagate=True,        # Let logs go to your main project (default)
    add_handler=False,     # Don't add separate handler (default)
)
```

### Independent package logging:

```python
from general_backend.logging.setup_logging import configure_package_logger

# Configure package logger to be independent
configure_package_logger(
    level=logging.DEBUG,
    propagate=False,       # Don't propagate to parent loggers
    add_handler=True,      # Add our own handler
    fmt='%(levelname)s: %(message)s'  # Custom format
)
```

## Best Practices

1. **As a dependency**: Do nothing! The package will work with your logging setup automatically.

2. **For fine control**: Use `configure_package_logger()` to adjust only the package's behavior.

3. **For standalone use**: Use `configure_standalone_logging()` to set up complete logging.

4. **For silencing noisy dependencies**: Use `set_logger_level_for_dependency()`.

## API Functions

The logging module provides these main functions:

- `get_logger()` - Get a package-scoped logger
- `configure_package_logger()` - Configure only the package logger (dependency-friendly)
- `configure_standalone_logging()` - Full logging setup for standalone use
- `set_logger_level_for_dependency()` - Control dependency package log levels