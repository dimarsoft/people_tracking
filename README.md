# people_tracking
В ходе изучения  фреймфорка FairMOT  было выполнено:

1. Разметка датасета для детекции совместно с остальными группами изучавшими встроенные трекеры YOLO и сторонние трекеры.
2. Конвертация лейблов формата YOLO - 0 0.616093 0.552211 0.234644 0.414005 где 0 - class_id 0.616093 - center_x 0.552211 - center_y 0.234644 - width 0.414005 - height
в формат FairMOT frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, Visibility где: 

frame – порядковый номер фрейма.

id - Идентификатор рамки (-1), id детектируемого объекта (Каждая траиектория объекта идентифицируется уникальным идентификатором (-1 для обнаружения).

bb_left -  Координаты x (это center_x YOLO * размер изображения х).

bb_top -  Координаты y (это center_y YOLO * размер изображения у).

bb_width -  Ширина (это width YOLO * ширину зображения / 2).

bb_height -  Высота (это height YOLO * высоту зображения / 2).

conf -  Область уверенности (0-1) (1) другими словами точность детектируемого объекта.

class -  класс объекта как в YOLO class_id.

Visibility -  Видимость или коэфициент видимости от 0 до 1, который указывает, какая часть этого объекта видна. Может быть из-за окклюзии или обрезки границ изображения.

  После конвертации и ряда эксперементов на базовом репозитории https://github.com/NS19972/FairMOT было принято решение переразметки т.к. имеющийся датасет не показывал видимых результатов в отслеживании объектов.

3. Было выполнено разбиение тренировочного видео с частотой до 30 кадров в секунду (частота обработки видео FairMOTом 26-30 кадров в сек), в ручную убраны пустые кадры и так же вручную поделены на тренировочную и валидационную выборки, идея заключалась в разметке трекингом (одного и того же идентификатора рамки) с помощью сервиса для разметки CVAT. Было размеченно порядка 9-10 тыс фреймов с общим количеством уникальных идентификаторов рамок = 297 (7 классов, Человек, Человек в каске, Человек в жилете, Человек в каске в жилете, Охранник, Каска, Жилет).
4. Были поставлены эксперементы с различными архитектурами базового репозитория https://github.com/NS19972/FairMOT (без новой разметки) и второстепенного репозитория с детектором YOLOv4 https://github.com/CaptainEven/YOLOV4_MCMOT на новой разметке.
5. Были изменены базовые файлы для генерации лейблов для последующего обучения сети (gen_labels_16.py).
