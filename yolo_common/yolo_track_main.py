"""
Модуль для запуска детекции и трекинга.
Функции для скачивания моделей.
"""
import json
from pathlib import Path
from typing import Union

import gdown

from configs import load_default_bound_line, WEIGHTS, YoloVersion, parse_yolo_version, ROOT, \
    get_all_optune_trackers, TEST_TRACKS_PATH, get_bound_line, get_detections_path
from tools.count_results import Result
# from tools.count_results import Result
from tools.exception_tools import print_exception
from post_processing.alex import alex_count_humans
from post_processing.timur import get_camera, timur_count_humans
from tools.resultools import results_to_json, TestResults, results_to_dict
from yolo_common.run_post_process import get_post_process_results
from yolo_common.yolo_detect import create_yolo_model
from yolo_common.yolo_track_bbox import YoloTrackBbox

folder_link = "https://drive.google.com/drive/folders/1b-tp_yxHgadeElP4XoDCFoXxCwXHK9CV"
yolo7_model_gdrive = "https://drive.google.com/drive/u/4/folders/1b-tp_yxHgadeElP4XoDCFoXxCwXHK9CV"
yolo7_model_gdrive_file = "25.02.2023_dataset_1.1_yolov7_best.pt"
yolo8_model_gdrive_file = "640img_8x_best_b16_e10.pt"

test_video_share_folder_link = \
    "https://drive.google.com/drive/folders/1YK0a3peuwdbvoZUAKciCvYM5KjKeizA6?usp=sharing"

test_video_share_folder_link_1_85 = \
    "https://drive.google.com/drive/folders/1o8xlmtVJkmvH7nDfFOQoBCoyRnhy9aVQ"

# для турникета
yolo8_turniket_model_gdrive_file = \
    "2023_05_12_17_52_39_yolo8_train_yolov8n.pt_epochs_30_batch_8_single_cls_best.pt"
yolo8_turniket_model_gdrive_link = \
    "https://drive.google.com/uc?id=1-dbgib8A6Tg4KLPbE2uZx7VdmAEUE7a0"


def get_local_path(yolo_version: YoloVersion) -> Path:
    """
    Получить путь к файлу модели на локальном диске
    Parameters
    ----------
    yolo_version
        Версия YOLO

    Returns
    -------
        Путь

    """
    if yolo_version == YoloVersion.yolo_v7:
        return Path(WEIGHTS) / yolo7_model_gdrive_file

    return Path(WEIGHTS) / yolo8_model_gdrive_file


def get_turniket_local_path(yolo_version: YoloVersion) -> Path:
    """
    Получить путь к файлу модели для детекции турникета на локальном диске
    Parameters
    ----------
    yolo_version
    Версия YOLO, поддерживается только 8

    Returns
    -------
    Путь

    """
    if yolo_version == YoloVersion.yolo_v7:
        raise Exception(f"{YoloVersion.yolo_v7} is not supported")

    return Path(WEIGHTS) / yolo8_turniket_model_gdrive_file


def get_link(yolo_version: YoloVersion) -> str:
    """
    Получить ссылку на гугл диск для скачивания модели
    Parameters
    ----------
    yolo_version
        Версия YOLO

    Returns
    -------
    Строку с ссылкой

    """
    if yolo_version == YoloVersion.yolo_v7:
        return 'https://drive.google.com/uc?id=1U6zt4rOy2v3VrLjMrsqdHF_Y6k9INbib'

    return 'https://drive.google.com/uc?id=1pyuTy4w1GPaPZKwJP9aKI0PlqTVW5xw9'


def get_model_file(yolo_version: YoloVersion) -> str:
    """
    Скачать и получить путь к модели Yolo
    :param yolo_version: Версия Yolo, 7,8
    :return: Путь к файлу модели
    """
    output_path = get_local_path(yolo_version)

    output = str(output_path)

    if output_path.exists():
        print(f"{output} local exist")
        return output

    url = get_link(yolo_version)

    print(f"download {output} from {url}")

    gdown.download(url, output, quiet=False)

    return output


def get_turniket_model_file(yolo_version: YoloVersion) -> str:
    """
    Скачать и получить путь к модели Yolo (для турникета)
    :param yolo_version: Версия Yolo, 7,8
    :return: Путь к файлу модели
    """
    output_path = get_turniket_local_path(yolo_version)

    output = str(output_path)

    if output_path.exists():
        print(f"{output} local exist")
        return output

    url = yolo8_turniket_model_gdrive_link

    print(f"download {output} from {url}")

    gdown.download(url, output, quiet=False)

    return output


def download_test_video():
    output = str(ROOT / "testinfo")

    url = 'https://drive.google.com/uc?id=1YK0a3peuwdbvoZUAKciCvYM5KjKeizA6'

    print(f"download {output} from {url}")

    folders = gdown.download_folder(id="1YK0a3peuwdbvoZUAKciCvYM5KjKeizA6",
                                    output=output, quiet=False)
    # folders = gdown.download_folder(url=url, quiet=False)
    print(folders)


def download_test_video_1_85(output: Union[str, Path]):
    output = str(output)

    folders = gdown.download_folder(id="1o8xlmtVJkmvH7nDfFOQoBCoyRnhy9aVQ",
                                    remaining_ok=True,
                                    output=output, quiet=False)
    print(folders)


def post_process(test_func, track: list, num, w, h, bound_line, source) -> Result:
    # count humans

    humans_result = Result(0, 0, 0, [])

    if test_func is not None:
        try:
            tracks_new = []
            for item in track:
                tracks_new.append([item[0], item[5], item[6], item[1], item[2],
                                   item[3], item[4], item[7]])

            if isinstance(test_func, str):

                humans_result = None

                if test_func == "popov_alex":
                    humans_result = alex_count_humans(tracks_new, num, w, h, bound_line)
                    pass
                if test_func == "timur":
                    humans_result = timur_count_humans(tracks_new, w, h, bound_line)
                    pass

            else:
                #  info = [frame_id,
                #  left, top,
                #  width, height,
                #  int(detection[4]), int(detection[5]), float(detection[6])]
                # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]
                # humans_result = test_func(tracks_new)
                # bound_line =  [[490, 662], [907, 613]]
                # num(str), w(int), h(int)

                humans_result = test_func(tracks_new, num, w, h, bound_line)

        except Exception as e:
            print_exception(e, "post processing")

    humans_result.file = str(Path(source).name)

    return humans_result


def run_single_video_yolo(source, yolo_info="7", conf=0.3, iou=0.45, test_func="timur",
                          tracker_type="fastdeepsort", log: bool = True) -> dict:
    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")
    model = get_model_file(yolo_version)

    reid_weights = str(Path(WEIGHTS) / "osnet_x0_25_msmt17.pt")

    num, w, h, fps = get_camera(source)

    print(f"num = {num}, w = {w}, h = {h}, fps = {fps}")

    model = create_yolo_model(yolo_version, model)

    # tracker_type = "fastdeepsort"
    # tracker_type = "ocsort"

    # all_trackers = get_all_trackers_full_path()
    all_trackers = get_all_optune_trackers()
    tracker_config = all_trackers.get(tracker_type)

    if log:
        print(f"tracker_type = {tracker_type}")

    track = model.track(
        source=source,
        conf_threshold=conf,
        iou=iou,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights,
        log=log
    )

    num, w, h, fps = get_camera(source)
    cameras_info = load_default_bound_line()
    bound_line = get_bound_line(cameras_info, num)

    humans_result = post_process(test_func, track, num, w, h, bound_line, source)

    test_result_file = TEST_TRACKS_PATH

    test_results = TestResults(test_result_file)

    test_results.add_test(humans_result)

    test_res = test_results.compare_to_file_v2(output_folder=None)

    res_dic = \
        {
            "result": json.loads(results_to_json(humans_result)),
            "test_result": test_res,
            "num": num,
            "width": w,
            "height": h,
            "fps": fps,
            "bound_line": bound_line,
            "file": str(source)
        }

    return res_dic


def get_results_video_yolo_txt(source, conf=0.3, test_func="timur",
                               tracker_type="fastdeepsort",
                               log: bool = True, ext: str = "txt") -> dict:
    txt_source_folder = str(get_detections_path())

    all_trackers = get_all_optune_trackers()
    tracker_config = all_trackers.get(tracker_type)

    if log:
        print(f"tracker_type = {tracker_type}")

    source_path = Path(source)

    txt_source = Path(txt_source_folder) / f"{source_path.stem}.{ext}"

    reid_weights = str(Path(WEIGHTS) / "osnet_x0_25_msmt17.pt")

    model = YoloTrackBbox()

    track = model.track(
        source=source,
        txt_source=txt_source,
        conf_threshold=conf,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights,
        log=log
    )

    num, w, h, fps = get_camera(source)
    cameras_info = load_default_bound_line()
    bound_line = get_bound_line(cameras_info, num)

    humans_result = get_post_process_results(test_func, track,
                                             num, w, h, fps, bound_line, source, log=True)

    res = json.loads(results_to_json(humans_result))
    res["fps"] = fps

    return res


def get_results_video_yolo(source, yolo_info="7", conf=0.3, iou=0.45, test_func="timur",
                           model: Union[str, Path, None] = None,
                           tracker_type="fastdeepsort",
                           tracker_config=None, log: bool = True) -> dict:
    num, w, h, fps = get_camera(source)

    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")

    if model is None:
        model = get_model_file(yolo_version)

    reid_weights = str(Path(WEIGHTS) / "osnet_x0_25_msmt17.pt")

    print(f"num = {num}, w = {w}, h = {h}, fps = {fps}")

    model = create_yolo_model(yolo_version, model)

    if tracker_config is None:
        all_trackers = get_all_optune_trackers()
        tracker_config = all_trackers.get(tracker_type)

    if log:
        print(f"tracker_type = {tracker_type}")

    track = model.track(
        source=source,
        conf_threshold=conf,
        iou=iou,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights,
        log=log
    )

    num, w, h, fps = get_camera(source)
    cameras_info = load_default_bound_line()
    bound_line = get_bound_line(cameras_info, num)

    humans_result = get_post_process_results(test_func, track, num, w, h, fps,
                                             bound_line, source, log=True)

    results = results_to_dict(humans_result)
    # res = json.loads(results_to_json(humans_result))
    results["fps"] = fps

    return results


if __name__ == '__main__':
    # download_test_video()

    # get_model_file(YoloVersion.yolo_v7)

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"

    # video_file = str(Path(video_source) / "1.mp4")
    video_file = "D:\\44.mp4"

    result = run_single_video_yolo(video_file, yolo_info="8ul", log=False)

    print(result)

    # res = Result(1, 2, 4, [])

    # print(res.__dict__)

    # str1 = results_to_json(res)
    # print(str1)

    # str2 = json.dumps(res.__dict__, indent=4)

    # print(str2)
