"""
Тестирование функций с помощью библиотеки pytest

https://pypi.org/project/pytest/
https://docs.pytest.org/en/latest/

"""
import pytest

from tools.count_results import Deviation
from tools.resultools import TestResults


@pytest.mark.parametrize(
    "dev1, dev2, expected",
    [
        pytest.param(Deviation(10, 100, 1), Deviation(10, 100, 1), True),
        pytest.param(Deviation(10, 100, 1), Deviation(0, 20, 1), True),
        pytest.param(Deviation(10, 100, 1), Deviation(210, 300, 1), False),
    ],
)
def test_dev(dev1: Deviation, dev2: Deviation, expected: bool):
    """
    Тестирование функции сравнения нарушения.
    Будет использован список параметров

    Parameters
    ----------
    dev1
    dev2
    expected

    Returns
    -------

    """
    assert TestResults.intersect_deviation(dev1, dev2) == expected
