import logging
import datetime
import os


def new_logger():
    logger = logging.getLogger("mainlogger")
    logger.setLevel(logging.INFO)

    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    log_filename = f"{current_time}.log"

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


logger = new_logger()
