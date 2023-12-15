import logging
import datetime

logger = logging.getLogger("mainlogger")
logger.setLevel(logging.INFO)

current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
log_filename = f"log/{current_time}.log"

file_handler = logging.FileHandler(log_filename, encoding="utf8")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s -%(levelname)s - %(message)s", "%H:%M:%S")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
