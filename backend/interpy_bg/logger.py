import logging

def get_console_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Creates a console logger with the given name and log level."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)

        # create console handler with same level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # formatter for console output
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # attach handler to logger
        logger.addHandler(console_handler)

        # optional: prevent logs from propagating to root logger
        logger.propagate = False

    return logger