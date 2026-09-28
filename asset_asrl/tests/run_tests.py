# -*- coding: utf-8 -*-

import importlib.util
import io
import logging
import sys
import unittest
from pathlib import Path

from pyflakes.api import checkPath
from pyflakes.reporter import Reporter


# %% Logging helpers

ANSI_RESET = "\033[0m"

ANSI_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[35;1m",
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

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = ColorFormatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_format)

    file_handler = logging.FileHandler("test_run.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger


logger = setup_logging()


# %% Pyflakes tests

class PyflakesTests(unittest.TestCase):
    """Verify Python source files contain no Pyflakes diagnostics."""

    def test_pyflakes(self):
        """Verify Python files contain no unused imports or variables."""

        root = Path(".").resolve()

        python_files = sorted(
            path for path in root.rglob("*.py")
            if ".git" not in path.parts
            and "__pycache__" not in path.parts
            and ".venv" not in path.parts
            and "venv" not in path.parts
            and "__init__" not in path.parts
        )

        self.assertGreater(len(python_files), 0, "No Python files found.")

        failures = []
        error_count = 0

        for path in python_files:
            stdout = io.StringIO()
            stderr = io.StringIO()
            reporter = Reporter(stdout, stderr)

            errors = checkPath(str(path), reporter)
            error_count += errors

            if errors:
                output = stdout.getvalue() + stderr.getvalue()
                failures.append(f"{path}:\n{output}")

        print(f"\nPyflakes errors: {error_count}")

        self.assertFalse(
            failures,
            "Pyflakes found issues:\n\n" + "\n".join(failures),
        )


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

    # Add Pyflakes test explicitly.
    suite.addTests(loader.loadTestsFromTestCase(PyflakesTests))

    test_files = sorted(start_dir.rglob(pattern))

    logger.info(f"Found {len(test_files)} test file(s).\n")

    for test_file in test_files:
        logger.info(f"Loading: {test_file}")

        module_name = f"_test_module_{test_file.stem}"

        try:
            spec = importlib.util.spec_from_file_location(module_name, test_file)

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