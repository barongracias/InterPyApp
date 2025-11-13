import time
import functools

def _get_logger_from_args(args):
    """Extract logger if args[0] is an object with .logger attribute."""
    if not args:
        return None
    obj = args[0]
    return getattr(obj, "logger", None)


def log_call(func):
    """Decorator that logs entry/exit using self.logger if available."""
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
    """Decorator that logs execution time using self.logger if available."""
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