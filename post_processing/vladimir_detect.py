from pathlib import Path
from typing import Union

from post_processing.timur import get_camera
from post_processing.vladimir import predict_model
from tools.count_results import Result, Deviation
from tools.resultools import results_to_dict


def vladimir_detect(source: Union[str, Path],
                    model_path: Union[str, Path, None] = None) -> dict:
    """
    Детектировать нарушения по видео
    :param source: путь к файлу видео.
    :param model_path: Путь к файлу модели YOLO, можно не указывать.
    :return: Словарь с результатами.
    """
    predict = predict_model(source=source, path_to_model=model_path)

    in_out = predict[0]
    counter_in = in_out['enter']
    counter_out = in_out['out']

    print('Вошло - ', counter_in)
    print('Вышло - ', counter_out)

    devs = predict[1]

    deviations = []

    for key, value in devs.items():
        print(f'Нарушитель с id-{key} ')
        for v in value:
            status_id = v[0]

            start_frame = v[1].split()[:-1]

            print(f'номер нарушения-{status_id}, момент нарушения -{start_frame}')

            deviations.append(Deviation(start_frame, start_frame, status_id))

    results = Result(counter_in + counter_out, counter_in, counter_out, deviations)
    results.file = str(source)

    results = results_to_dict(results)

    num, w, h, fps = get_camera(source)

    results["fps"] = fps

    return results
