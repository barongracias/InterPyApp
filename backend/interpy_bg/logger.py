# imports
import logging
import os

def get_console_logger(name: str, directory: str) -> logging.Logger:
    """
    Creates a console logger with the given name and log level.
    
    Args:
        name (str): Name of the logger (typically __name__ of the module).
        directory (str): Directory to store log files. Must exist or be created.
    Returns:
        logging.Logger: Conffigured logger instance.
    
    """
    
    if not directory:
        raise ValueError("log_dir must be provided")
    
    # ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    logger = logging.getLogger(name.split('.')[-1])
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # create console handler with same level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # formatter for console output
        formatter = logging.Formatter(
            fmt='%(asctime)s - (%(name)s) - [%(levelname)s]: %(message)s',
            datefmt='%d/%m/%y %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # file handler
        filename = os.path.join(directory, f"{name.split('.')[-1]}.log")
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - (%(name)s) - [%(levelname)s]: %(message)s',
            datefmt='%d/%m/%y %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # prevent logs from propagating to root logger
        logger.propagate = False

    return logger