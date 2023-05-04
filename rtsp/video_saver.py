import argparse
from pathlib import Path
from typing import Union
from datetime import datetime
import cv2

from tools.exception_tools import print_exception


def create_file_name(tag: str, w: int, h: int, fps: int, file_num: int = 0) -> str:
    now = datetime.now()

    session_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                   f"{now.second:02d}_{now.microsecond}_{file_num}"

    return f"{tag}_{session_name}_{w}_{h}_fps_{fps}.mp4"


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

        frames_per_file = fps * 5 * 60  # 5мин
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

                    session_name = create_file_name(tag, w, h, fps, files)

                    output_video_path = output_folder / session_name

                    output_video = cv2.VideoWriter(
                        str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (w, h))

                    frames = 0

                frames += 1
                print(f"frames = {frames}")

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

    # rtsp_capture_to_file("rtsp://stream:ShAn675sb5@31.173.67.209:13554",
    #                     tag="31_173_67_209_13554",
    #                     output_folder="c:\\AI\\rtsp\\", )
    opt = parser.parse_args()
    # print(opt)

    run_cli(opt)
