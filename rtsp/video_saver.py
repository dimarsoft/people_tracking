import argparse
import time
from threading import Thread
from pathlib import Path
from queue import Queue
from typing import Union
from datetime import datetime
import cv2

from main import run_group1_detection
from rtsp.rtsp_stream_queue_to_file import RtspStreamReaderToFile, init_cv, print_timed
from tools.exception_tools import print_exception
from tools.path_tools import create_file_name


def run_det(source: str) -> Thread:
    thread = Thread(target=run_detect(source))
    thread.daemon = True
    thread.start()

    return thread


def run_detect(source: str):
    print(f"start process {source}")
    res = 0  # run_group1_detection(source)

    print(f"end process {source}")

    print(res)


def rtsp_capture_to_file_2(rtsp_url: str, tag: str, output_folder: Union[str, Path]) -> None:
    """
    Запись rtsp потока в файлы по 5 мин
    Parameters
    ----------
    rtsp_url Адрес камеры
    tag Тэг камеры, название
    output_folder Куда пишем

    Returns
    -------

    """

    try:

        init_cv()

        reader = RtspStreamReaderToFile(rtsp_url=rtsp_url, tag=tag, output_folder=output_folder)

        print_timed(f"{__name__}, reader start")
        reader.start()

        while True:

            print_timed(f"{__name__}, start sleep")

            time.sleep(1000)

            print_timed(f"{__name__}, stop sleep")

            break
            ## reader.stop()

            # if cv2.waitKey(1) == ord("q"):
            #    break

        print_timed(f"{__name__}, reader stop")
        reader.stop()

    except Exception as ex:
        print_exception(ex, "rtsp_capture_to_file")


def rtsp_capture_to_file(rtsp_url: str, tag: str, output_folder: Union[str, Path]) -> None:
    """
    Запись rtsp потока в файлы по 5 мин
    Parameters
    ----------
    rtsp_url Адрес камеры
    tag Тэг камеры, название
    output_folder Куда пишем

    Returns
    -------

    """
    input_video = None

    threads: list[Thread] = []

    try:

        print(f"connect to {rtsp_url}")

        input_video = cv2.VideoCapture(rtsp_url)

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"connected to {rtsp_url} : fps = {fps}, w = {w}, h ={h}")

        output_folder = Path(output_folder)

        session_name = create_file_name(tag, w, h, fps)

        output_video_path = output_folder / session_name

        output_video = cv2.VideoWriter(
            str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h))

        frames = 0

        files = 0

        frames_per_file = fps * 1 * 10  # 5мин
        while True:
            try:
                ret, frame = input_video.read()
            except Exception as ex:
                print_exception(ex, "read frame")
                continue

            if ret:
                if frames > frames_per_file:
                    output_video.release()
                    files += 1

                    # th = run_det(str(output_video_path))

                    # threads.append(th)

                    session_name = create_file_name(tag, w, h, fps, files)

                    output_video_path = output_folder / session_name

                    output_video = cv2.VideoWriter(
                        str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (w, h))

                    frames = 0

                frames += 1

                if frames % (fps * 10) == 0:
                    print(f"file = {files}: frames = {frames}")

                output_video.write(frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            else:
                print(f"frame not read")

            if files > 20:
                break

        output_video.release()
    except Exception as ex:
        print_exception(ex, "rtsp_capture_to_file")

    if input_video is not None:
        input_video.release()

    for x in threads:
        x.join()


# запуск из командной строки: python video_server.py  --output_folder "."
def run_cli(opt_info):
    rtsp_capture_to_file(rtsp_url=opt_info.rtsp_url,
                         tag=opt_info.tag,
                         output_folder=opt_info.output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rtsp_url', type=str, default="rtsp://stream:ShAn675sb5@31.173.67.209:13554",
                        help='rtsp адрес')
    parser.add_argument('--tag', type=str, default="31_173_67_209_13554",
                        help='тэг/имя камеры')  # file/folder, 0 for webcam
    parser.add_argument('--output_folder', type=str, help='output_folder')  # output folder

    rtsp_capture_to_file_2("rtsp://stream:ShAn675sb5@31.173.67.209:13554",
                           tag="31_173_67_209_13554",
                           output_folder="c:\\AI\\rtsp\\", )
    opt = parser.parse_args()
    # print(opt)

    # run_cli(opt)
