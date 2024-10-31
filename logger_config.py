import logging
import os

# Logging levels
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()

# Reset logger.log before running the script
if os.path.exists("logger.log"):
    os.remove("logger.log")
file_handler = logging.FileHandler("logger.log")

# Archive all logs in logger_archive.log
file_handler_archive = logging.FileHandler("logger_archive.log")

# Set level and format for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
file_handler_archive.setLevel(logging.INFO)

# Define the formatter without microseconds
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
file_handler_archive.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.addHandler(file_handler_archive)

# Clear the contents of file_handler
# with open("logger.log", "w"):
#     pass
