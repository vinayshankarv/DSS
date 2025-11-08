import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

# ---------------------------------------------------------------------
# Resolve paths safely even when run from subdirectories (e.g., via Streamlit)
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name: str, log_file: str = "scraper.log"):
    """
    Creates and configures a rotating logger that writes to /logs/scraper.log.
    Works regardless of where the script is called from (main dir or subdir).
    """

    log_path = os.path.join(LOG_DIR, log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # DEBUG → INFO → WARNING → ERROR → CRITICAL

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # File handler (1 MB max, 3 backups)
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Startup message
    logger.info(f"Logger initialized for module: {name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return logger


# ---------------------------------------------------------------------
# Self-test (manual run only)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_logger = setup_logger("test_logger")
    test_logger.info("✅ Logger test started.")
    test_logger.warning("⚠️ This is a sample warning.")
    test_logger.error("❌ This is a test error.")
    test_logger.debug("🐍 Debug log working fine.")
