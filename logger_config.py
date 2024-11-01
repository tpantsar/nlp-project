import logging
import sys

# Logging levels
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

PRINT_TO_CONSOLE = True
PRINT_TO_FILE = True

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers with UTF-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("logger.log", encoding="utf-8")
file_handler_archive = logging.FileHandler("logger_archive.log", encoding="utf-8")

# Reset logger.log before running the script
with open("logger.log", "w", encoding="utf-8"):
    pass

# Set level and format for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
file_handler_archive.setLevel(logging.INFO)

# Create a formatter and set the format for the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

if PRINT_TO_CONSOLE:
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

if PRINT_TO_FILE:
    file_handler.setFormatter(formatter)
    file_handler_archive.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(file_handler_archive)
