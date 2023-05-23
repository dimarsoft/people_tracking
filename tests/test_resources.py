"""
Тестирование функций с помощью встроенного модуля unittest
http://docs.python.org/library/unittest.html
"""

import unittest

from tools.change_bboxes import no_change_bbox
from tools.count_results import Deviation
from tools.resultools import TestResults
from tools.save_txt_tools import yolo7_save_tracks_to_txt, yolo_load_detections_from_txt


class FunctionsTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_no_change_bbox(self):
        """
        Проверка функции no_change_bbox.
        Изменений быть не должно!

        Returns
        -------

        """
        bbox = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        test_bbox = no_change_bbox(bbox, file_id="1")

        for i in range(4):
            self.assertEqual(test_bbox[i], bbox[i],
                             f"{i} 'элемент изменился. Эх no_change_bbox не работает корректно, "
                             "доработать!!!")

    def test_compare_dev(self):
        """
        Тестирование функции сравнения нарушений
        Returns
        -------

        """

        dev1 = Deviation(10, 100, 1)

        self.assertEqual(True, TestResults.intersect_deviation(dev1, dev1),
                         "сам себя же должен пересекать")

        dev2 = Deviation(0, 20, 1)

        self.assertEqual(True, TestResults.intersect_deviation(dev1, dev2),
                         "ну вот провал")

        dev3 = Deviation(101, 200, 1)
        self.assertEqual(False, TestResults.intersect_deviation(dev1, dev3))
        self.assertEqual(False, TestResults.intersect_deviation(dev2, dev3))

    def test_text_save_empty(self):
        """
        Проверка записи/чтения пустого списка
        Returns
        -------

        """
        file_to_save = "test_empty.txt"
        yolo7_save_tracks_to_txt([], file_to_save)

        from_file = yolo_load_detections_from_txt(file_to_save)

        self.assertEqual(0, len(from_file))

    def test_text_save(self):
        """
        Запись/чтение списка из вух элементов
        Returns
        -------

        """
        file_to_save = "test.txt"
        tracks = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8]
        ]

        yolo7_save_tracks_to_txt(tracks, file_to_save)

        from_file = yolo_load_detections_from_txt(file_to_save)

        self.assertEqual(2, len(from_file))


if __name__ == '__main__':
    unittest.main()
