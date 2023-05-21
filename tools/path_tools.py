"""
Модуль для работы с путями
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_video_files(source, files: Optional[list[str]]) -> list[str]:
    """
    Получить список файлов
    Parameters
    ----------
    source
    files

    Returns
    -------

    """
    # список файлов с видео для обработки
    list_of_videos = []

    source_path = Path(source)

    if source_path.is_dir():
        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                if files is None:
                    list_of_videos.append(str(entry))
                else:
                    if entry.stem in files:
                        list_of_videos.append(str(entry))

    else:
        list_of_videos.append(str(source))

    return list_of_videos


def create_session_folder(yolo_version, output_folder, task: str) -> str:
    """
    Создание имени папки для сессии
    Parameters
    ----------
    yolo_version
        Версия Юлы
    output_folder
        Папка для сессии
    task
        Название задачи

    Returns
    -------

    """
    now = datetime.now()

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_"\
                          f"{now.minute:02d}_" \
                          f"{now.second:02d}_{yolo_version}_{task}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    return str(session_folder)


def get_log_time_str() -> str:
    """
    Строка с датой для логирования
    Returns
    -------

    """
    now = datetime.now()
    return f"{now.day:02d}-{now.month:02d}-{now.year:04d} {now.hour:02d}:{now.minute:02d}:" \
           f"{now.second:02d}"


def get_tag_time_str() -> str:
    """
    Текущая дата и время в строку
    Returns
    -------

    """
    now = datetime.now()
    return f"{now.day:02d}_{now.month:02d}_{now.year:04d}_{now.hour:02d}_{now.minute:02d}_" \
           f"{now.second:02d}"


def get_time_str(dt_start: datetime, dt_end: datetime) -> str:
    """
    Даты в строку
    Parameters
    ----------
    dt_start
    dt_end

    Returns
    -------

    """
    return f"{dt_start.day:02d}_{dt_start.month:02d}_{dt_start.year:04d}_{dt_start.hour:02d}_"\
           f"{dt_start.minute:02d}_" \
           f"{dt_start.second:02d}_"\
           f"{dt_end.hour:02d}_{dt_end.minute:02d}_" \
           f"{dt_end.second:02d}"


def create_file_name(tag: str, width: int, height: int, fps: int, file_num: int = 0,
                     ext: str = "mp4") -> str:
    """
    Создать имя файла
    Parameters
    ----------
    tag
        Заголовок
    width
        Ширина
    height
        Высота
    fps
        FPS
    file_num
        Номер файла
    ext
        Расширение файла

    Returns
    -------

    """
    now = datetime.now()

    session_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_"\
                   f"{now.minute:02d}_" \
                   f"{now.second:02d}_{now.microsecond}_{file_num}"

    return f"{tag}_{session_name}_{width}_{height}_fps_{fps}.{ext}"
