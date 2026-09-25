"""Standard-library logging configured once for the project namespace."""

from __future__ import annotations

import logging

from business_entity_resolution.config.settings import LoggingSettings

_LOGGER_NAME = "business_entity_resolution"
_HANDLER_MARKER = "_business_entity_resolution_console_handler"


def configure_logging(settings: LoggingSettings) -> logging.Logger:
    """Configure readable console logging without changing the global root logger."""

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(settings.level)
    logger.propagate = False

    formatter = logging.Formatter(settings.format, datefmt=settings.date_format)
    handler = next(
        (
            existing
            for existing in logger.handlers
            if getattr(existing, _HANDLER_MARKER, False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    handler.setLevel(settings.level)
    handler.setFormatter(formatter)
    return logger


def get_logger(module_name: str | None = None) -> logging.Logger:
    """Return a logger in the project namespace for a module or component."""

    if not module_name or module_name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if module_name.startswith(f"{_LOGGER_NAME}."):
        return logging.getLogger(module_name)
    return logging.getLogger(f"{_LOGGER_NAME}.{module_name}")
