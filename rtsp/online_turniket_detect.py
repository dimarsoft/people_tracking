import time
from pathlib import Path
from typing import Union

from rtsp import init_cv, print_timed
from rtsp.RtspStreamTurniketDetect import RtspStreamTurniketDetect
from tools.exception_tools import print_exception


def rtsp_capture_to_detect(rtsp_url: str, tag: str, output_folder: Union[str, Path]) -> None:
    try:
        # инициализация CV и логирования
        init_cv()

        reader = RtspStreamTurniketDetect(rtsp_url=rtsp_url, tag=tag,
                                          output_folder=output_folder)

        print_timed(f"{__file__}, reader start")
        reader.start()

        print_timed(f"{__file__}, start sleep")

        time.sleep(60)

        print_timed(f"{__file__}, stop sleep")

        print_timed(f"{__file__}, reader stop")

        reader.stop()

    except Exception as ex:
        print_exception(ex, "rtsp_capture_to_file")


if __name__ == '__main__':
    # rtsp_url_m = "C:\\AI\\2023\\corridors\\dataset-v1.1\\test\\20.mp4"
    rtsp_url_m = "rtsp://stream:ShAn675sb5@31.173.67.209:13554"

    rtsp_capture_to_detect(rtsp_url_m,
                           tag="31_173_67_209_13554",
                           output_folder="c:\\AI\\rtsp\\images", )
