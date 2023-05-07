import logging
import os
import sys
import traceback
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Union, Optional
import time
import cv2

from tools.path_tools import create_file_name, get_log_time_str


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
    set_logging(filename=filename)
    th = min(os.cpu_count(), 8)
    cv2.setNumThreads(th)


class RtspStreamReaderToFile(object):
    def __init__(self, rtsp_url: str, tag: str, output_folder: Union[str, Path], time_split: int = 5):
        """
        Класс для чтения из потока Rtsp и запись в папку.
        :param rtsp_url: Строка подключения к rtsp.
        :param tag: Наименование камеры.
        :param output_folder: Папка, в которую пишем.
        :param time_split: Деление файлов по протяженности, сек. Если -1 не делим.
        """
        self.rtsp_url = rtsp_url
        self.tag = tag
        self.output_folder = Path(output_folder)
        self.time_split = time_split
        self.frames_per_file = -1
        self._queue = Queue()
        self._stream_reader: Optional[Thread] = None  # Создается и запускается в Start
        self._file_writer: Optional[Thread] = None  # Останавливается в Stop

        self._started = False
        self._stop = False
        self._stopWriter = False
        self._startedWriter = False

        # информация о параметрах видео потока, будет получена в первом подключении
        self._video_info_valid = False
        self._fps = 0
        self._width = 0
        self._height = 0

        self.output_video = None
        self.files = 0

    def start(self):
        if self._started:
            return
        self._stop = False
        self._video_info_valid = False

        self._stream_reader = Thread(target=self._stream_reader_proc)
        self._file_writer = Thread(target=self._file_writer_proc)

        self._stream_reader.start()
        self._start_writer()

        self._started = True

        print_timed(f"threads started, {self.rtsp_url}")

    def stop(self):
        if not self._started:
            print_timed(f"threads not started, {self.rtsp_url}", is_error=True)
            return

        print_timed(f"threads stopping, {self.rtsp_url}")

        self._stop = True
        _started = False

        self._stream_reader.join()
        self._stop_writer()

        print_timed(f"threads stopped, {self.rtsp_url}")

    def _init_info(self, video_stream: cv2.VideoCapture):
        self._fps = int(video_stream.get(cv2.CAP_PROP_FPS))
        # ширина
        self._width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        self._height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.time_split > 0:
            self.frames_per_file = self._fps * self.time_split * 60  # минуты в кол-во фреймов
        else:
            self.frames_per_file = -1

        self._video_info_valid = True

        print_timed(f"{self.rtsp_url} : fps = {self._fps}, w = {self._width}, "
                    f"h = {self._height}, frames_per_file = {self.frames_per_file}")

    def _stream_reader_proc(self):

        print_timed(f"start _stream_reader_proc {self.rtsp_url}")

        is_connected = False

        input_video = None

        while not self._stop:
            if not is_connected:
                print_timed(f"connect to {self.rtsp_url}")
                input_video = cv2.VideoCapture(self.rtsp_url)

                is_connected = input_video.isOpened()

                if is_connected:
                    print_timed(f"connected to {self.rtsp_url}")

                    if not self._video_info_valid:
                        self._init_info(input_video)
            else:
                is_connected = input_video.isOpened()
                if not is_connected:
                    print_timed(f"disconnected from {self.rtsp_url}")
                    input_video.release()
                    continue

            try:
                ret, frame = input_video.read()
            except Exception as ex:
                print_exception_time(ex, "read frame")
                continue

            if ret:
                self._queue.put(frame)

        input_video.release()

        print_timed(f"reader finished from {self.rtsp_url}")

    def _start_writer(self):
        if self._startedWriter:
            print_timed(f"writer already started, {self.rtsp_url}", is_error=True)
            return

        self._startedWriter = True
        self._file_writer.start()
        print_timed(f"_start_writer = {self.rtsp_url}")

    def _stop_writer(self):
        if not self._startedWriter:
            print_timed(f"writer not started, {self.rtsp_url}", is_error=True)
            return

        self._stopWriter = True
        self._file_writer.join()
        self._startedWriter = False

        print_timed(f"_stop_writer: {self.rtsp_url}")

    def _close_write_video(self):
        if self.output_video is not None:
            self.output_video.release()
            self.output_video = None

    def _get_write_video(self, force_create: bool = False):
        if force_create:
            self._close_write_video()

        if self.output_video is not None:
            return self.output_video

        session_name = create_file_name(self.tag, self._width, self._height, self._fps, self.files)

        output_video_path = self.output_folder / session_name

        print_timed(f"start new file: {output_video_path}, {self.rtsp_url} ")

        self.output_video = cv2.VideoWriter(
            str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'),
            self._fps, (self._width, self._height))

        return self.output_video

    def _file_writer_proc(self):
        frames = 0

        last_frame_time = None

        while not self._stopWriter:

            if not self._video_info_valid:
                continue

            if self._queue.empty():
                # делаем проверку, что долго не было фреймов и закрываем файл
                if last_frame_time is not None:
                    secs = int(elapsed_sec(last_frame_time))

                    if secs > 100:
                        self._close_write_video()
                        last_frame_time = None
                        print_timed(f"no frames {secs} s")
                continue

            frame = self._queue.get()

            if frame is not None:
                last_frame_time = time_synch()
                if self.time_split > 0 and frames > self.frames_per_file:
                    self.files += 1
                    output_video = self._get_write_video(force_create=True)

                    frames = 0
                else:
                    output_video = self._get_write_video(force_create=False)

                frames += 1

                if frames % (self._fps * 10) == 0:
                    print_timed(f"file = {self.files}: frames = {frames}")

                output_video.write(frame)

        self._close_write_video()
