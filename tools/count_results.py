"""
Ответ имеет структуру словаря:
{
    "file": 'файл видео',
    "fps": 'fps видео',
    "counter_in" : целое число,     # число вошедших
    "counter_out" : целое число     # число вышедших
    "deviations":                   # нарушения
    [
        {
            "start_frame": целое число,
            "end_frame": целое число,
            "status_id": код нарушения  # 1, 2, 3: см. get_status()
        }
    ]
}
"""


class Deviation(object):
    def __init__(self, start, end, status):
        self.start_frame = int(start)
        self.end_frame = int(end)
        self.status_id = int(status)

    def __str__(self):
        return f"{self.status_id}: [{self.start_frame}, {self.end_frame}]"


class Result:
    def __init__(self, humans, c_in, c_out, deviations: list[Deviation]):
        self.file = ""
        self.humans = int(humans)
        self.counter_in = int(c_in)
        self.counter_out = int(c_out)
        self.deviations = deviations

    def __str__(self):
        return f"file = {self.file}, in = {self.counter_in}, " \
               f"out = {self.counter_out}, deviations = {len(self.deviations)}"


def get_status(status) -> str:
    """
    Строковое представление типа нарушения

    :param status: тип нарушения
    :return: Строка
    """
    status = int(status)
    if status == 1:
        return "без каски и жилета"
    if status == 2:
        return "без жилета"
    if status == 3:
        return "без каски"
    return ""


def from_status(status_id: int) -> tuple[int, int]:
    """
    Возвращает информацию о каске и жилете по коду нарушения
    :param status_id: Код нарушения
    :return: пара чисел каска, жилет
    1 - нарушение, т.е. нет жилета или каски
    """
    if status_id == 0:
        return 0, 0
    if status_id == 1:
        return 1, 1
    if status_id == 2:
        return 0, 1
    return 1, 0
