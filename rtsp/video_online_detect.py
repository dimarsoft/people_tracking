import json
import time
from pathlib import Path
from typing import Union

from rtsp import print_timed, RtspStreamReaderToDetect
from rtsp.rtsp_tools import init_cv
from tools.exception_tools import print_exception
from tools.save_txt_tools import yolo7_save_tracks_to_txt
from tools.video_tools import VideoInfo
from yolo_common.run_post_process import get_post_process_results
from yolo_common.yolo8_online import SaveResultsBase


class SaveResults(SaveResultsBase):

    def __init__(self, output_folder: Union[str, Path]):
        self.output_folder = Path(output_folder)
        self.output_file = self.output_folder / "rtsp_results.log"
        with open(self.output_file, "w") as write_file:
            pass

    def update(self, results, start_id, end_id, video_info: VideoInfo):
        message = f"start_id = {start_id}, end_id = {end_id}, results = {len(results)}'\n'"

        with open(self.output_file, "a") as write_file:
            write_file.write(message)

        bound_line = [
            [
                int(video_info.width * 294 / 640),
                int(video_info.height * 384 / 640)
            ],
            [
                int(video_info.width * 416 / 640),
                int(video_info.height * 352 / 640)
            ]
        ]
        result_tracks_file = self.output_folder / f"{start_id}_track_results.json"
        yolo7_save_tracks_to_txt(results=results, txt_path=result_tracks_file)

        humans_result = get_post_process_results("timur", results,
                                                 1,
                                                 video_info.width, video_info.height, video_info.fps,
                                                 bound_line, "source", log=True)

        result_json_file = self.output_folder / f"{start_id}_track_results.json"
        print(f"Save result_items '{str(result_json_file)}'")
        with open(result_json_file, "w") as write_file:
            write_file.write(json.dumps(humans_result, indent=4, default=lambda o: o.__dict__))


def rtsp_capture_to_detect(rtsp_url: str, tag: str, output_folder: Union[str, Path]) -> None:
    try:
        # инициализация CV и логирования
        init_cv()

        saver = SaveResults(output_folder)

        reader = RtspStreamReaderToDetect(rtsp_url=rtsp_url, tag=tag,
                                          output_folder=output_folder, saver=saver)

        print_timed(f"{__file__}, reader start")
        reader.start()

        print_timed(f"{__file__}, start sleep")

        time.sleep(6000)

        print_timed(f"{__file__}, stop sleep")

        print_timed(f"{__file__}, reader stop")

        reader.stop()

    except Exception as ex:
        print_exception(ex, "rtsp_capture_to_file")


if __name__ == '__main__':
    rtsp_url = "C:\\AI\\2023\\corridors\\dataset-v1.1\\test\\8.mp4"
    # rtsp_url = "rtsp://stream:ShAn675sb5@31.173.67.209:13554"

    rtsp_capture_to_detect(rtsp_url,
                           tag="31_173_67_209_13554",
                           output_folder="c:\\AI\\rtsp\\", )
