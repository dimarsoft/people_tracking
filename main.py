__version__ = 1.2

import argparse
from pathlib import Path
from typing import Union, Optional

import torch

from post_processing.group_1_detect import group_1_detect
from post_processing.vladimir_detect import vladimir_detect
from utils.torch_utils import date_modified, git_describe, git_branch_name
from yolo_common.yolo_track_main import get_results_video_yolo


def print_version():
    git_info = git_describe()

    if git_info is None:
        git_info = f"{date_modified()}"
    else:
        git_info = f"git: {git_info}, {date_modified()}"

    version = f'Version: {__version__}, {git_info}, torch {torch.__version__}'  # string
    print(f"Humans, helmets and uniforms. {version}")


def get_version(version=__version__, path=__file__) -> str:
    git_info = git_describe()
    branch_name = git_branch_name()

    if git_info is None:
        git_info = f"{date_modified(path)}"
    else:
        git_info = f"git: {branch_name}:{git_info}, {date_modified()}"

    return f'{version}, {git_info}, torch {torch.__version__}'


def create_empty_dict(video_source) -> dict:
    results = \
        {
            "file": str(video_source),
            "counter_in": 0,
            "counter_out": 0,
            "deviations": []
        }

    return results


def run_group1_detection(video_source: Union[str, Path],
                         model: Union[str, Path, None] = None) -> dict:
    """
    Вызов обработки видео файла кодом группы №1.
    :param video_source: Путь к файлу.
    :param model: Путь к модели, можно не указывать, тогда загрузится с гуглдиска.
    :return: Возвращаем словарь с результатом
    """
    return group_1_detect(video_source, model_path=model)


def run_group2_detection(video_source: Union[str, Path],
                         model: Union[str, Path, None] = None,
                         tracker_name: Optional[str] = None,
                         tracker_config: Union[str, dict, Path, None] = None,
                         log: bool = False) -> dict:
    if tracker_name is None:
        tracker_name = "fastdeepsort"
    return get_results_video_yolo(video_source,
                                  model=model,
                                  tracker_type=tracker_name,
                                  tracker_config=tracker_config,
                                  log=log)


def run_vladimir_detection(video_source: Union[str, Path],
                           model: Union[str, Path, None] = None) -> dict:
    return vladimir_detect(video_source,
                           model_path=model)


def run_detection(version: Union[str, int],
                  source: Union[str, Path],
                  model: Union[str, Path, None] = None,
                  tracker_name: Optional[str] = None,
                  tracker_config: Union[str, dict, Path, None] = None) -> Optional[dict]:
    if version in (1, "1"):
        return run_group1_detection(source, model=model)

    if version in (2, "2"):
        return run_group2_detection(source, model=model,
                                    tracker_name=tracker_name, tracker_config=tracker_config)

    if version in (3, "3"):
        return run_vladimir_detection(source, model=model)

    return None


def run_cli(opt_info) -> dict:
    """
    Запуск через командную строку
    Parameters
    ----------
    opt_info
        Параметры командной строки

    Returns
    -------
    Словарь с результатом
    """
    version, source, tracker_name, tracker_config, model = \
        opt_info.version, opt_info.source, opt_info.tracker_name, opt_info.tracker_config, opt_info.model

    return run_detection(version=version, source=source,
                         tracker_name=tracker_name, tracker_config=tracker_config,
                         model=model)


if __name__ == '__main__':
    print_version()

    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str,
                        help='version of post processing: "1" - group 1, "2" - group 2, 3 - Владимир')  #
    parser.add_argument('--source', type=str, help='path to video file')  #
    parser.add_argument('--model', type=str, default=None, help='path to yolo model')  #
    parser.add_argument('--tracker_name', type=str, default=None, help='tracker_name')
    parser.add_argument('--tracker_config', type=str, default=None, help='tracker_config: dict, file')

    opt = parser.parse_args()
    print(opt)
    print(run_cli(opt))
    # для проверки
    # file_video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\3.mp4"
    # print(run_vladimir_detection(file_video_source))
