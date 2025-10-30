import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

print(">>> Logger file loaded. __name__ =", __name__)

load_dotenv(override=False)

def setup_logger():
    # Detect environment
    env = os.getenv("APP_ENV", "local").lower()
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Setup log directory and file
    try:
        log_dir = Path(__file__).resolve().parents[1] / "logs"
        log_dir.mkdir(exist_ok=True)
    except Exception as e:
    # Fallback if Docker container doesn’t have write permission
        print(f"[Logger Warning] Could not create log directory: {e}")
        log_dir = Path("/tmp")
    log_file = log_dir / f"app_{env}.log"


     # ---- Remove existing handlers to prevent duplication ----
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)


    # Define consistent format
    log_format = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(funcName)s:%(lineno)d - %(message)s"
    )
    formatter = logging.Formatter(log_format)

   
    # ---- File Handler (rotating) ----
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # ---- Console Handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # ---- Base Logger Config ----
    root_logger.setLevel(log_level)

    # In Docker, also send logs to stdout (so visible via `docker logs`)
    # This is handled automatically by StreamHandler, so we just need to ensure it’s active
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # ---- Named Logger ----
    logger = logging.getLogger("gold_forecasting_api")
    logger.info(f"Logger initialized for environment: {env} | Level: {log_level_str}")

    return logger