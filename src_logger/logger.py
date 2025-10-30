import logging
import os
from logging.handlers import RotatingFileHandler
import sys
import json

class JsonFormatter(logging.Formatter):
    """Custom formatter for JSON structured logs (for production)."""
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def get_logger(name: str):
    """Central logger setup with environment awareness."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Prevent duplicate handlers

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    env = os.getenv("ENV", "development").lower()

    logger.setLevel(log_level)

    # Console handler (always present)
    console_handler = logging.StreamHandler(sys.stdout)

    if env == "production":
        console_handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler for local/debug only
    if env == "development":
        file_handler = RotatingFileHandler(
            "logs/app.log", maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
