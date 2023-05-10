import logging
import os
import sys
import time
import traceback
from datetime import datetime

import cv2


def get_log_time_str() -> str:
    now = datetime.now()
    return f"{now.day:02d}-{now.month:02d}-{now.year:04d} {now.hour:02d}:{now.minute:02d}:" \
           f"{now.second:02d}"


def create_file_name(tag: str, w: int, h: int, fps: int, file_num: int = 0, ext: str = "mp4") -> str:
    now = datetime.now()

    session_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                   f"{now.second:02d}_{now.microsecond}_{file_num}"

    return f"{tag}_{session_name}_{w}_{h}_fps_{fps}.{ext}"


def time_synch():
    return time.time()


def elapsed_sec(t: time):
    return (time_synch() - t) * 100


def print_timed(message: str, is_error: bool = False):
    message = f"{get_log_time_str()} : {message}"
    print(message)

    if is_error:
        logging.error(message)
    else:
        logging.info(message)


def print_exception_time(e: Exception, caption: str):
    print_timed(f"Exception {e} in {caption}!!!", is_error=True)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)


def set_logging(rank=-1, filename: str = "rtsp.log"):
    logging.basicConfig(
        filename=filename,
        format="[%(levelname)s][%(thread)d] %(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_cv(filename: str = "rtsp.log"):
    """
    Настройка логирования и CV
    :param filename: Путь к файлу лога
    :return:
    """
    set_logging(filename=filename)
    th = min(os.cpu_count(), 8)
    cv2.setNumThreads(th)
