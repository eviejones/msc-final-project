import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Creates a configured logger with consistent formatting across modules.

    Args:
        name (str): Logger name (typically shown in log output).
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The configured logger.
    """
    logging.basicConfig(level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
