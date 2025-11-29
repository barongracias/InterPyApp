# Lightweight logger for fivedreg (mirrors interpy_bg.logger)
import logging
import os


def get_console_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """
    Return a console logger with optional file logging.

    Args:
        name: Logger name.
        log_dir: Optional directory to write log files.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - (%(name)s) - [%(levelname)s]: %(message)s', datefmt='%d/%m/%y %H:%M:%S')

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # file handler if requested
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
