import unittest
import sys
import logging
from pathlib import Path

# ANSI escape codes for colors
ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    logging.DEBUG: "\033[36m",    # Cyan
    logging.INFO: "\033[32m",     # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",    # Red
    logging.CRITICAL: "\033[35;1m" # Bright Magenta
}

class ColorFormatter(logging.Formatter):
    """Custom logging formatter with ANSI colors for console output."""
    def format(self, record):
        color = ANSI_COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{ANSI_RESET}"

def setup_logging():
    """Configure logging with colored console output and plain file logging."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)

    # File handler without colors
    file_handler = logging.FileHandler("test_run.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

def run_all_tests(start_dir=Path("."), pattern="test_*.py"):
    """
    Discover and run all unittests starting from `start_dir` matching `pattern`.
    """
    start_dir = Path(start_dir).resolve()
    logger.info(f"Starting test discovery in: {start_dir} (pattern: {pattern})")

    if not start_dir.is_dir():
        logger.error(f"Directory '{start_dir}' does not exist.")
        sys.exit(1)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(start_dir), pattern=pattern)
    test_count = suite.countTestCases()
    logger.info(f"Discovered {test_count} test(s).")

    if test_count == 0:
        logger.warning("No tests found. Exiting.")
        sys.exit(0)

    logger.info("Running tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")

    sys.exit(not result.wasSuccessful())

if __name__ == "__main__":
    default_dir = Path("tests") if Path("tests").is_dir() else Path(".")
    run_all_tests(start_dir=default_dir, pattern="test_*.py")
