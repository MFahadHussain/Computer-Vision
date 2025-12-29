"""Logger configuration and setup."""

import logging

logger = logging.getLogger(__name__)


def setup_logger(level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

