import os
import json
import logging
import logging.config
from pathlib import Path
from functools import wraps


def setup_logging(
    console: bool = True,
    level: str = "WARNING",
    configuration: str = "logging.json",
    env_key="LOG_CFG",
):
    """Setup logging configuration"""
    if (value := os.getenv(env_key, None)) is not None:
        configuration = value
    configuration = Path(configuration).resolve()
    if configuration.is_file():
        with configuration.open("rt") as f:
            logging.config.dictConfig(json.loads(f), level=level)
    else:
        logging.basicConfig(level=level)

    logging.captureWarnings(True)


def logged_method_call(the_method, msg="call"):
    @wraps(the_method)
    def _wrapper(self, *args, **kwargs):
        logger = logging.getLogger(the_method.__module__)
        logger.debug(f"{self.__class__.__name__}#{id(self)}.{the_method.__name__}: {msg}")
        return the_method(self, *args, **kwargs)

    _wrapper.__doc__ = the_method.__doc__
    return _wrapper


def logged_function_call(the_function, msg="call"):
    @wraps(the_function)
    def _wrapper(*args, **kwargs):
        logger = logging.getLogger(the_function.__module__)
        logger.debug(f"{the_function.__name__}: {msg}")
        return the_function(*args, **kwargs)

    return _wrapper


def set_log_level(value: str) -> None:
    """Update the logging level."""
    valid = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if value not in valid:
        raise ValueError(f"`log_level` is invalid. Please use one of: {valid}")
    logging.getLogger().setLevel(value)
