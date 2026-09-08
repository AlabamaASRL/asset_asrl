# -*- coding: utf-8 -*-

import importlib.util
import logging
import sys
import unittest
from pathlib import Path


# %% Logging helpers

# ANSI escape codes for colors

# Could be abstracted to create custom logging class for use with rest of ASSET

ANSI_RESET = "\033[0m"

ANSI_COLORS = {
    logging.DEBUG: "\033[36m",       # Cyan
    logging.INFO: "\033[32m",        # Green
    logging.WARNING: "\033[33m",     # Yellow
    logging.ERROR: "\033[31m",       # Red
    logging.CRITICAL: "\033[35;1m",  # Bright Magenta
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

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

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

    # Prevent log propagation to root logger
    logger.propagate = False

    return logger


logger = setup_logging()


# %% Unit test runner

def run_all_tests(start_dir=Path("tests"), pattern="test_*.py"):
    """
    Discover and run all unittests starting from `start_dir` matching `pattern`.

    Test directories do not need to contain `__init__.py` files.
    """

    start_dir = Path(start_dir).resolve()

    logger.info(
        f"Starting test discovery in: {start_dir} "
        f"(pattern: {pattern})\n"
    )

    if not start_dir.is_dir():
        logger.error(f"Directory '{start_dir}' does not exist.")
        sys.exit(1)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_files = sorted(start_dir.rglob(pattern))

    logger.info(f"Found {len(test_files)} test file(s).\n")

    for test_file in test_files:
        logger.info(f"Loading: {test_file}")

        module_name = f"_test_module_{test_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                test_file,
            )

            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {test_file}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)

        except Exception:
            logger.exception(f"Failed to load test file: {test_file}")

    test_count = suite.countTestCases()
    logger.info(f"\nDiscovered {test_count} test(s).\n")

    if test_count == 0:
        logger.warning("No tests found. Exiting.")
        sys.exit(0)

    logger.info("Running tests...\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    logger.info(f"\nTests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")

    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    run_all_tests(start_dir=Path("."), pattern="test_*.py")