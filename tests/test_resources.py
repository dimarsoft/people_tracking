# http://docs.python.org/library/unittest.html
import unittest

from tools.change_bboxes import no_change_bbox


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


if __name__ == '__main__':
    unittest.main()
