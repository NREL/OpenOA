__version__ = "2.3"
"""
When bumping version, please be sure to also update parameters in sphinx/conf.py
"""

import os
import json
import logging
import logging.config


def setup_logging(default_path="logging.json", default_level=logging.INFO, env_key="LOG_CFG"):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


setup_logging()


def logged_method_call(the_method, msg="call"):
    def _wrapper(self, *args, **kwargs):
        logger = logging.getLogger(the_method.__module__)
        logger.debug(
            "{}#{}.{}: {}".format(self.__class__.__name__, id(self), the_method.__name__, msg)
        )
        return the_method(self, *args, **kwargs)

    _wrapper.__doc__ = the_method.__doc__
    return _wrapper


def logged_function_call(the_function, msg="call"):
    def _wrapper(*args, **kwargs):
        logger = logging.getLogger(the_function.__module__)
        logger.debug("{}: {}".format(the_function.__name__, msg))
        return the_function(*args, **kwargs)

    return _wrapper
