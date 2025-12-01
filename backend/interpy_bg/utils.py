"""Utility decorators for logging and instrumentation used across the interpy_bg package."""

import time
import functools

def _get_logger_from_args(args):
    """
    Attempt to retrieve a ``logger`` attribute from the first positional argument.

    Args:
        args: Positional arguments passed to the wrapped function.

    Returns:
        A logger-like object if the first argument exposes a ``logger`` attribute,
        otherwise ``None``.
    """
    if not args:
        return None
    obj = args[0]
    return getattr(obj, "logger", None)


def log_call(func):
    """
    Decorator that logs entry/exit for a function using ``self.logger`` when available.

    The decorator inspects the first positional argument (e.g., ``self`` on instance
    methods) for a ``logger`` attribute and emits debug messages on entry/exit.

    Args:
        func: Callable to wrap.

    Returns:
        Callable with identical behaviour to ``func`` but with optional logging side
        effects when a logger is present.
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
        Callable that returns the same value as ``func`` while logging elapsed
        milliseconds if a logger is available.
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
