"""Lightweight logger utilities mirroring the logging behaviour in other backends."""

import logging
import os


def get_console_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """
    Create or return a configured console logger with optional file output.

    Args:
        name: Identifier used when retrieving or creating the logger.
        log_dir: Directory to persist log files to disk; when ``None`` only console
            output is configured.

    Returns:
        A ``logging.Logger`` configured with a console handler and, if requested,
        a file handler writing to ``<log_dir>/<name>.log``.
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
