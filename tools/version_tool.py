__version__ = 0.1

import torch
import ultralytics

from utils.torch_utils import git_describe, git_branch_name, date_modified


def get_version(version=__version__, path=__file__) -> str:
    """
    Получить строку с версией ПО + информация о git, torch и ultralytics

    Parameters
    ----------
    version
        Версия
    path
        Путь к файлу, чтобы узнать дату его изменения

    Returns
    -------
        Стройка с полной версией ПО

    """
    git_info = git_describe()
    branch_name = git_branch_name()

    if git_info is None:
        git_info = f"{date_modified(path)}"
    else:
        git_info = f"git: {branch_name}:{git_info}, {date_modified()}"

    return f'{version}, {git_info}, torch = {torch.__version__}, ultralytics = {ultralytics.__version__}'
