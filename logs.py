import logging
import sys
import datetime
import os.path


def get_logger(name: str, print_level=logging.DEBUG):
    formatter = logging.Formatter(
        fmt="%(levelname)6s [%(filename)15s:%(lineno)-3d %(asctime)s] %(message)s",
        datefmt='%H:%M:%S',
    )
    time_now = datetime.datetime.now().strftime("%Y_%m_%d.%H_%M_%S")
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    if not os.path.exists(f'log/{time_now}/'):
        os.makedirs(f'log/{time_now}/', exist_ok=True)
    file_handler = logging.FileHandler(f'log/{time_now}/{name}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger
