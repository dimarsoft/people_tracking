"""
Тестирование функций с помощью встроенного модуля unittest
http://docs.python.org/library/unittest.html
"""


import unittest

from tools.change_bboxes import no_change_bbox
from tools.count_results import Deviation
from tools.resultools import TestResults


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


if __name__ == '__main__':
    unittest.main()
