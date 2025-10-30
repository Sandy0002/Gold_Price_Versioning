import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def setup_logger():
    env = os.getenv("APP_ENV", "local").lower()
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()

    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"app_{env}.log"

    if env in ["local", "development"]:
        level = logging.DEBUG
        
        
    elif env in ["docker", "staging"]:
        level = logging.INFO
    else: # production
        level = logging.WARNING

    log_format = (
        "%(asctime)s | %(levelname)s | %(name)s | "
        "%(funcName)s:%(lineno)d - %(message)s"
    )

    # Prevent duplicate handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])

    logger = logging.getLogger("gold_forecasting_api")
    logger.info(f"Logger initialized for environment: {env}")
    return logger
