import sys
import traceback
from pathlib import Path
from typing import Union


def save_exception(e: Exception, text_ex_path: Union[str, Path], caption: str) -> None:
    """
    Сохранение информации об исключении в текстовый файл
    Parameters
    ----------
    e
        Объект исключения
    text_ex_path
        Путь к файлу
    caption
        Заголовок сообщения

    Returns
    -------

    """
    with open(text_ex_path, "w") as write_file:
        write_file.write(f"Exception in {caption}!!!")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        for line in lines:
            write_file.write(line)
    print(f"Exception in {caption} {str(e)}! details in {str(text_ex_path)} ")


def print_exception(e: Exception, caption: str) -> None:
    """
    Вывод на экран сообщения об ошибке/исключении
    Parameters
    ----------
    e
        Объект исключения
    caption
        Заголовок сообщения

    Returns
    -------

    """
    print(f"Exception {e} in {caption}!!!")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for line in lines:
        print(line)
