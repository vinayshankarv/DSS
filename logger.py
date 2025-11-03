import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime


def setup_logger(name: str, log_file: str = "scraper.log"):
    """
    Purpose:
    - Creates and configures a logger that writes to /logs/scraper.log
    - Includes timestamp, log level, module name, and message.
    - Supports rotation (max 1MB, 3 backups).

    Args:
        name (str): Logger name (usually module name, e.g., "flipkart", "amazon", "save_to_csv")
        log_file (str): Log file name. Default = scraper.log
    """

    # Ensure /logs directory exists
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", log_file)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # DEBUG → INFO → WARNING → ERROR → CRITICAL

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Create rotating file handler (1 MB max per file, 3 backups)
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")

    # Console handler for quick view
    console_handler = logging.StreamHandler()

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


if __name__ == "__main__":
    # Self-test
    test_logger = setup_logger("test_logger")
    test_logger.info("✅ Logger test started.")
    test_logger.warning("⚠️ This is a sample warning.")
    test_logger.error("❌ This is a test error.")
    test_logger.debug("🐍 Debug log working fine.")
