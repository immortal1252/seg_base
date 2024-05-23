import logging
import datetime
import os.path


def new_logger(log_dir=""):
    logger = logging.getLogger("mainlogger")
    logger.setLevel(logging.DEBUG)

    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    log_filename = f"{current_time}.log"

    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode="w", encoding="utf8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger
