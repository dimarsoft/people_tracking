__version__ = 0.1

import torch

from utils.torch_utils import git_describe, git_branch_name, date_modified


def get_version(version=__version__, path=__file__) -> str:
    git_info = git_describe()
    branch_name = git_branch_name()

    if git_info is None:
        git_info = f"{date_modified(path)}"
    else:
        git_info = f"git: {branch_name}:{git_info}, {date_modified()}"

    return f'{version}, {git_info}, torch {torch.__version__}'
