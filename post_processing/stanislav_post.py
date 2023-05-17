from pandas import DataFrame

from post_processing.stanislav import Human_f
from tools.count_results import Result
from tools.exception_tools import print_exception


def convert_track_to_df(tracks: list, w: int, h: int) -> DataFrame:
    """
    Конвертация в DataFrame нужный постобработке
    Args:
        tracks: Список треков
        w: Ширина фрейма(картинки)
        h: Высота

    Returns:
        DataFrame(columns=['frame', 'id', 'left', 'top', 'width', 'height', 'cl']
    """
    new_track = []
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for item in tracks:
        cls = item[2]
        bbox_left, bbox_top, bbox_w, bbox_h = item[3] * w, item[4] * h, item[5] * w, item[6] * h

        new_item = [int(item[0]), int(item[1]), int(bbox_left), int(bbox_top), int(bbox_w), int(bbox_h), int(cls)]

        new_track.append(new_item)

    tracker_data = DataFrame(new_track, columns=['frame', 'id', 'left', 'top', 'width', 'height', 'cl'])

    return tracker_data


# пример от Станислава
def stanislav_count_humans(tracks: list, num, w, h, bound_line, log: bool = True) -> Result:
    # Турникет Станислава, потом нужно перейти на общий

    turniket_dict: dict[int, int] = \
        {1: 470, 2: 470, 3: 470, 4: 910, 5: 470,
         6: 580, 7: 580, 8: 470, 9: 470, 10: 470,
         11: 470, 12: 470, 13: 470, 14: 580, 15: 470,
         16: 470, 17: 470, 18: 470, 19: 470, 20: 850,
         21: 430, 22: 430, 23: 430, 24: 430, 25: 510,
         26: 510, 27: 510, 28: 510, 29: 510, 30: 510,
         31: 430, 32: 0, 33: 0, 34: 0, 35: 0,
         36: 430, 37: 0, 38: 430, 39: 430, 40: 430,
         41: 0, 42: 430, 43: 0
         }
    f = int(num)

    turnic_y = turniket_dict.get(f)

    if turnic_y is None:
        pt1 = bound_line[0]
        pt2 = bound_line[1]
        turnic_y = int((pt1[1] + pt2[1]) / 2)

    tracker_data = convert_track_to_df(tracks, w, h)

    try:
        # подсчет входов и выходов (людей)
        r_in, r_out, passlist = Human_f(turnic_y, tracker_data)
        return Result(r_in + r_out, r_in, r_out, [])
    except Exception as ex:
        print_exception(ex, "Human_f")

    # ошибка будет в print_exception, но результат вернем
    return Result(0, 0, 0, [])
