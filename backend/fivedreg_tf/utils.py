"""Utility decorators for logging and instrumentation used across the fivedreg_tf package."""

import time
import functools


def _get_logger_from_args(args):
    """
    Attempt to retrieve a `logger` attribute from the first positional argument.

    Args:
        args: Positional arguments passed to the wrapped function.

    Returns:
        A logger-like object if the first argument exposes a `logger` attribute, otherwise ``None``.
    """
    if not args:
        return None
    obj = args[0]
    return getattr(obj, "logger", None)

def log_call(func):
    """
    Decorator that emits entry/exit debug logs around the wrapped callable.

    The decorator looks for a `logger` attribute on the first positional argument
    (typically ``self`` on instance methods). When present, it logs the function
    name when entering and exiting the function.

    Args:
        func: Callable to wrap.

    Returns:
        The wrapped callable that behaves identically to ``func`` but logs around it
        when a logger is available.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = _get_logger_from_args(args)
        if logger:
            logger.debug(f"[CALL] → {func.__name__}()")
        result = func(*args, **kwargs)
        if logger:
            logger.debug(f"[CALL] ← {func.__name__}()")
        return result
    return wrapper

def timer(func):
    """
    Decorator that measures execution time and logs it via ``self.logger`` when present.

    Args:
        func: Callable to time.

    Returns:
        The wrapped callable that returns the same result as ``func`` while emitting
        a debug log with elapsed milliseconds if a logger is available.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = _get_logger_from_args(args)
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000  # ms
        if logger:
            logger.debug(f"[TIMER] {func.__name__} took {elapsed:.2f} ms")
        return result
    return wrapper
