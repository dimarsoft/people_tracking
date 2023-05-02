from numpy import ndarray
from ultralytics import YOLO
import numpy as np
import pandas as pd
from IPython.display import clear_output
import cv2
import os
from sklearn.metrics import precision_recall_fscore_support

glob_kwarg = {'barier': 337, 'tail_mark': False, 'tail': 200, 're_id_mark': False, 're_id_frame': 11,
              'tail_for_count_mark': False, 'tail_for_count': 200, 'two_lines_buff_mark': False, 'buff': 40,
              'go_men_forward': False, 'step': 45, 'height': 100}

ocsort_kwarg = {'det_thresh': 0.49428431641933235, 'max_age': 7,
                'min_hits': 7, 'iou_threshold': 0.6247364818234254,
                'delta_t': 5, 'asso_func': 'iou',
                'inertia': 0.6758806605183052, 'use_byte': True}  # default


def get_boxes(result):  # эта функция сохраняет боксы от предикта в файл .npy для того что бы не возвращаться больше к детекции
    all_boxes = np.empty((0, 7))
    for i, r in enumerate(result):
        bbox = r.cpu().boxes.data.numpy()
        bbox = np.hstack((bbox, np.tile(i, (bbox.shape[0], 1))))
        all_boxes = np.vstack((all_boxes, bbox))
        orig_shp = r.orig_shape
    return all_boxes, orig_shp


def detect_videos(path_model, model_in_path, video_source, start_vid=1, end_vid=1):
    if end_vid == 1:
        length = len([f for f in os.listdir(video_source)
                      if f.endswith('.mp4') and os.path.isfile(os.path.join(video_source, f))])  # подсчитаем количество видео в папке
    else:
        length = end_vid
    for N in range(start_vid, length+1):  # устанавливаем какие видео смотрим
        try:
            with open(video_source + f'{N}.mp4', 'r') as f:
                # каждый раз инициализируем модель в колабе иначе выдает ошибочный результат
                model = YOLO(path_model+model_in_path)
                all_boxes, orig_shape = get_boxes(model.predict(source=video_source + f'{N}.mp4',
                                                                line_thickness=2, vid_stride=1, stream=True, save=False))
                np.save(
                    path_model + f"{N}.npy", np.array((orig_shape, all_boxes), dtype=object))
        except:
            print(f'Видео {N}: отсутствует')


# Если обрезаем бокс (функцию можно модифицировать по необходимости проверки той или иной гипотезы)
def change_bbox(bbox, **glob_kwarg):
    tail = glob_kwarg['tail']
    y2 = bbox[:, [1]] + tail
    bbox[:, [3]] = y2
    return bbox


def forward(bbox, tracks,
            fwd=False):  # эта функция позволяет сохранить лист детекций в который внесены айди от трека сохраняя нетрекованные боксы на случай последующей перетрековки
    # Создадим пустой массив для каждого кадра который будем наполнять
    person = np.empty((0, 8))
    for i, bb in enumerate(bbox):  # Сравним каждый первичный не треккованный бокс
        for k, t in enumerate(tracks):  # С каждым треккованым
            if round(t[0]) == round(bb[0]) and round(t[1]) == round(bb[1]) and round(t[2]) == round(bb[2]) and round(
                    t[3]) == round(bb[3]):  # Если у них совпадают координаты
                bb_tr = np.copy(bb)
                # Добавляем к нетрекованному боксу трек определенный треккером (таким образом сохраняя конфиденс и класс)
                bb_tr = np.insert(bb_tr, 6, t[4])
                # Складываем массив. На этом этапе остались в стеке только трекованные боксы. Но нам хотелось бы сохранить их все для фиксации нарушений или последующей перетрековки
                person = np.vstack((person, bb_tr))
            else:
                pass
        if fwd:
            # добавим в оттрекованный массив то что треккер отсеял (на случай перетрековки)
            if sum(np.in1d(bb[:4], tracks[:, :4])) < 4:
                person = np.vstack((person, np.insert(bb, 6, -1)))

    return person


def tracking_on_detect(all_boxes, tracker, orig_shp, **glob_kwarg) -> ndarray:
    """
    эта функция отправляет нетрекованные боксы людей в треккер прокидывая
    остальные классы мимо треккера
    :param all_boxes:
    :param tracker:
    :param orig_shp:
    :return:
    """
    tail_mark = glob_kwarg['tail_mark']
    all_boxes_tr = np.empty((0, 8))
    if len(all_boxes) != 0:
        for i in range(int(max(all_boxes[:, -1]))):
            bbox = all_boxes[all_boxes[:, -1] == i]
            # отбираем форму и каски в отдельный массив который прокинем мимо трека
            bbox_unif = bbox[np.where(bbox[:, 5] != 0)][:, :6]
            # добавляем столбец с айди нан для касок и жилетов
            bbox_unif = np.hstack(
                (bbox_unif, np.tile(np.nan, (bbox_unif.shape[0], 1))))
            # сохраняем номер кадра
            bbox_unif = np.hstack(
                (bbox_unif, np.tile(i, (bbox_unif.shape[0], 1))))
            bbox = bbox[np.where(bbox[:, 5] == 0)]  # в трек идут только люди
            if tail_mark:  # Если обрезаем бокс
                bbox = change_bbox(bbox, **glob_kwarg)
            tracks = tracker.update(
                bbox[:, :-2], img_size=orig_shp, img_info=orig_shp)  # трекуем людей
            # эта функция позволяет использовать далее лист детекций в который внесены айди от трека (трек фильтрует и удаляет боксы)
            person = forward(bbox, tracks, fwd=False)
            # складываем людей в массив
            all_boxes_tr = np.vstack((all_boxes_tr, person))
            # складываем каски и жилеты в массив
            all_boxes_tr = np.vstack((all_boxes_tr, bbox_unif))
    return all_boxes_tr


# функция отрисовки боксов на соответсвующем видео
def create_video_with_bbox(bboxes, video_source, video_out):
    '''Функция записывает видео с рамками объектов, которые передаются в:
  bboxes - ndarray(x1, y1, x2, y2, conf, class, id, frame),
  если последовательность нарушена надо менять внутри функции.
  Другие обязательные аргументы функции:
  video_source - полный путь до исходного видео, на которое нужно наложить рамки;
  video_out - полный путь вновь создаваемого видео. Путь должен быть,
  а файла - не должно быть'''
    vid_src = cv2.VideoCapture(video_source)
    if vid_src.isOpened():
        # Разрешение кадра
        frame_size = (int(vid_src.get(3)), int(vid_src.get(4)))
        # Количество кадров в секунду
        fps = int(vid_src.get(5))
        # Количество кадров в файле
        len_frm = int(vid_src.get(7))
        # Выходное изображение записываем
        vid_out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'),
                                  fps, frame_size)
        # Пройдемся по всем кадрам
        for i in range(len_frm):
            ret, frame = vid_src.read()
            # На всякий пожарный случай выход
            if not ret:
                break
            # Отбираем рамки для кадра
            bbox = bboxes[bboxes[:, -1] == i, :-1]
            if len(bbox) > 0:
                # Только люди
                pbox = bbox[bbox[:, 5] == 0]
                for p in pbox:
                    # Добавим рамки
                    x1, y1 = int(p[0]), int(p[1])
                    x2, y2 = int(p[2]), int(p[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (153, 153, 153), 2)
                    # Добавим надпись в виде идентификатора объекта  и conf
                    msg = 'id' + str(int(p[6])) + ' ' + str(round(p[4], 2))
                    (w, h), _ = cv2.getTextSize(
                        msg, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - 20),
                                  (x1 + w, y1), (153, 153, 153), -1)
                    cv2.putText(frame, msg, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.45,
                                (255, 255, 255), 1)
                # Отрисовываем рамки касок
                helmets = bbox[bbox[:, 5] == 1]
                for helmet in helmets:
                    x1, y1 = int(helmet[0]), int(helmet[1])
                    x2, y2 = int(helmet[2]), int(helmet[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 255, 204), 2)
                # Отрисовываем рамки жилетов
                vests = bbox[bbox[:, 5] == 2]
                for vest in vests:
                    x1, y1 = int(vest[0]), int(vest[1])
                    x2, y2 = int(vest[2]), int(vest[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 204, 255), 2)
            # Записываем кадр
            vid_out.write(frame)
        # Освобождаем видео
        vid_out.release()
        vid_src.release()
    else:
        print('Видеофайл недоступен')
    # Заклинание cv2
    cv2.destroyAllWindows()


def find_iou(cover_coords, man_coords):
    x1, y1, x2, y2 = cover_coords
    x3, y3, x4, y4 = man_coords
    if x1 >= x4 or x3 >= x2 or y1 >= y4 or y3 >= y2:
        return 0
    int1x, int1y = max(x1, x3), max(y1, y3)
    int2x, int2y = min(x2, x4), min(y2, y4)
    area_int = (int2x - int1x) * (int2y - int1y)
    area_1 = (x2 - x1) * (y2 - y1)
    area_2 = (x4 - x3) * (y4 - y3)
    return area_int / (area_1 + area_2 - area_int)


def find_intersection(cover_coords, man_coords):
    x1, y1, x2, y2 = cover_coords
    x3, y3, x4, y4 = man_coords
    if x1 >= x4 or x3 >= x2 or y1 >= y4 or y3 >= y2:
        return 0
    int1x, int1y = max(x1, x3), max(y1, y3)
    int2x, int2y = min(x2, x4), min(y2, y4)
    int_area = (int2x - int1x) * (int2y - int1y)
    cover_area = (x2 - x1) * (y2 - y1)
    return int_area / cover_area


def get_men(out_boxes, helmet_limit=0.5, vest_limit=0.5, metric=find_intersection):
    men = np.empty((0, 9))
    frames = int(max(out_boxes[:, -1])) if out_boxes.shape[0] > 0 else 0
    for i in range(frames + 1):
        # Только люди
        humans = out_boxes[(out_boxes[:, -3] == 0) &
                           (out_boxes[:, -2] != -1) & (out_boxes[:, -1] == i)]
        # Только каски
        helmets = out_boxes[(out_boxes[:, -3] == 1) & (out_boxes[:, -1] == i)]
        # Только жилетки
        vests = out_boxes[(out_boxes[:, -3] == 2) & (out_boxes[:, -1] == i)]

        if metric:
            helmet_men = []  # Создаем список айди людей в касках
            for cover in helmets:  # Заполняем список
                for man in humans:
                    if metric(cover[:4], man[:4]) > helmet_limit and man[-2] not in helmet_men:
                        helmet_men.append(man[-2])
                        continue

            vest_men = []  # Создаем список айди людей в жилетах
            for cover in vests:  # Заполняем список
                for man in humans:
                    if metric(cover[:4], man[:4]) > vest_limit and man[-2] not in vest_men:
                        vest_men.append(man[-2])
                        continue

            # Персональный подход к каждому человеку
            for man in humans:
                # Есть ли на человеке каска
                helmet = 0 if man[-2] in helmet_men else 1
                # Есть ли на человеке жилет
                vest = 0 if man[-2] in vest_men else 1
                # Это условие нужно для начала работы, потому что несколько первых кадров
                # не имеют идентификатора. Можно отбросить первые кадры, здесь так и делается
                # Это просто добавление в массив men. Часть параметров нужны нам дважды для выявления макс и мин.
                # Поэтому дважды повторяются ордината низа и номер кадра
                men = np.vstack((men, np.array([man[-2],
                                                man[1], man[1],
                                                man[3], man[3],
                                                i, i,
                                                helmet, vest])))
            # Формируем датафрейм. Будем считать низ вначале и в конце, также первый кадр
            # и последний кадр, где был этот id (пока это одно и тоже значение)

        if not metric:
            inter_helm = 90  # поле вокруг бб человека для детекции касок
            inter_unif = 90  # поле вокруг бб человека для детекции жилетов
            # Персональный подход к каждому человеку
            for man in humans:
                # Сколько касок в пределах рамок человека (либо 1, либо 0)
                helmet = 0 if len(helmets[(helmets[:, 0] >= (man[0] - inter_helm)) & (helmets[:, 1] >= (man[1] - inter_helm)) &
                                          (helmets[:, 2] <= (man[2] + inter_helm)) & (helmets[:, 3] <= (man[3] + inter_helm))]) >= 1 else 1
            # Сколько жилеток в пределах рамок человека (либо 1, либо 0)
                vest = 0 if len(vests[(vests[:, 0] >= (man[0] - inter_unif)) & (vests[:, 1] >= (man[1] - inter_unif)) &
                                      (vests[:, 2] <= (man[2] + inter_unif)) & (vests[:, 3] <= (man[3] + inter_unif))]) >= 1 else 1
            # Это условие нужно для начала работы, потому что несколько первых кадров
            # не имеют идентификатора. Можно отбросить первые кадры, здесь так и делается
                # Это просто добавление в массив men. Часть параметров нужны нам дважды для выявления макс и мин.
                # Поэтому дважды повторяются ордината низа и номер кадра
                men = np.vstack((men, np.array([man[-2],
                                                man[1], man[1],
                                                man[3], man[3],
                                                i, i,
                                                helmet, vest])))
    return men


# эта длинная функция по сути работает как фильтр людей(пропускает дальше тех кто либо вошел либо вышел)
def get_count_men(men, orig_shape, **glob_kwarg):
    n_ = int(max(men[:, 0]))
    orig_shape = int(orig_shape)

    barier = glob_kwarg['barier']
    re_id_mark = glob_kwarg['re_id_mark']
    re_id_frame = glob_kwarg['re_id_frame']
    tail_for_count_mark = glob_kwarg['tail_for_count_mark']
    tail_for_count = glob_kwarg['tail_for_count']
    two_lines_buff_mark = glob_kwarg['two_lines_buff_mark']
    buff = glob_kwarg['buff']
    go_men_forward = glob_kwarg['go_men_forward']

    format = orig_shape / 640

    gate_y = barier * format  #

    box_y_top = [None] + [list() for _ in range(int(n_))]
    box_y_bottom = [None] + [list() for _ in range(int(n_))]
    box_frame = [None] + [list() for _ in range(int(n_))]

    for i, m in enumerate(men):
        id = int(m[0])
        frame_n = int(m[-4])
        box_frame[id].append(frame_n)

        if re_id_mark:  # Если  значение True  то переназначаем айди принудительно в случае отсутсвия айди более чем re_id_frame кадров
            y_top = float(m[2])
            y_bottom = float(m[4])
            box_y_top[id].append(y_top)
            box_y_bottom[id].append(y_bottom)

            # принудительно переназначаем айди на входе и выходе если треккер не переназначил (актуально для непотимизированных треккеров)
            if len(box_frame[id]) > re_id_frame:
                condition = box_frame[id][-1] - box_frame[id][-2] > 10 and (
                    (box_y_top[id][-1] / orig_shape < 0.2) or (box_y_bottom[id][-1] / orig_shape > 0.8))
                # условие смены id:
                # верхняя граница бб в верхней части кадра  или нижняя граница бб в нижней части кадра
                # бб не детектировался в течение 20 предыдущих кадров
                if condition:
                    n_ += 1
                    box_y_top.append([])
                    box_y_bottom.append([])
                    box_frame.append([])
                    for j in range(i, len(men)):
                        if men[j][0] == id:
                            men[j][0] = n_

    # if len(box_frame[id]) < 21: # удаляем айди чей трек короче n кадров
    #   men = men[~np.isin(men[:,0], id)]

    n = int(max(men[:, 0]))
    human_c = [None] + [list() for _ in range(int(n))]
    for m in men:
        num = int(m[0])
        box_center = (float(m[4]) - float(m[2]))/2+float(m[2])  # центр масс

        if tail_for_count_mark:  # Если  значение True  то считаем не центр масс а y_top + tail_for_count
            if float(m[2]) + tail_for_count * format < 640 * format:
                box_center = float(m[2]) + tail_for_count * format
            else:
                box_center = 640 * format

        human_c[num].append(box_center)

    ind = []
    for i, h in enumerate(human_c):

        if two_lines_buff_mark:  # если стоит маркер считать по одной из двух линий
            if h and h[0] < gate_y < h[-1]:
                ind.append(i)
            elif h and h[0] < (gate_y-buff) < h[-1]:
                ind.append(i)
            elif h and h[0] > gate_y > h[-1]:
                ind.append(i)
            elif h and h[0] > (gate_y-buff) > h[-1]:
                ind.append(i)

        if not two_lines_buff_mark:  # Считаем по одной линии
            if h and h[0] < gate_y < h[-1]:
                ind.append(i)
            elif h and h[0] > gate_y > h[-1]:
                ind.append(i)

    if not go_men_forward:
        # Здесь при False (default) дальше в массиве остаются только те айди
        # которые зафиксировались как вошедшие или вышедшие
        # (этот же маркер используется в следующей функции при подсчете нарушений)
        men = men[np.isin(men[:, 0], ind)]

    return men


def get_count_vialotion(men, orig_shape, **glob_kwarg):
    orig_shape = int(orig_shape)
    format = orig_shape/640

    go_men_forward = glob_kwarg['go_men_forward']
    step = glob_kwarg['step']
    height = glob_kwarg['height']

    df = pd.DataFrame(men, columns=['id',
                                    'first_top_y', 'last_top_y',
                                    'first_bottom_y', 'last_bottom_y',
                                    'first_frame', 'last_frame',
                                    'helmet', 'uniform'])
    # Зададим правила агрегирующих функций
    agg_func = {'first_top_y': 'first', 'last_top_y': 'last',
                'first_bottom_y': 'first', 'last_bottom_y': 'last',
                'first_frame': 'first', 'last_frame': 'last',
                'helmet': 'mean', 'uniform': 'mean'
                }
    # Группируем по id
    df1 = df.groupby('id').agg(agg_func)
    # Чтобы не выводилось предупреждение
    pd.options.mode.chained_assignment = None

    # Определяем количество вошедших и вышедших
    if go_men_forward:
        # Если  mark True то центр масс считается расстоянием от y_top. Если при этом координата центра масс больше формата то она приравнивается формату
        df1.loc[:, 'last_cent_y_stand'] = df1.last_top_y+height*format
        # здесь считаем низ от макушки а не по низу бокса, фиксируем, но если он ниже размера картинки то лимитируем по низу картинки
        df1.loc[df1['last_cent_y_stand'] > orig_shape,
                "last_cent_y_stand"] = orig_shape
        df1.loc[:, 'first_cent_y_stand'] = df1.first_top_y+height*format
        # здесь считаем низ от макушки а не по низу бокса, фиксируем, но если он ниже размера картинки то лимитируем по низу картинки
        df1.loc[df1['first_cent_y_stand'] > orig_shape,
                "first_cent_y_stand"] = orig_shape
        df1.loc[:, 'distance'] = df1.last_bottom_y - df1.first_bottom_y
        # здесь если человек вошел к турникету и вернулся ко входу то его дистанция будет около нуля и он отсеется, также как и охранник в обратном случае
        df2 = df1.loc[(np.abs(df1.distance) > (step*format))]

    if not go_men_forward:
        # Если  mark False то просто считаем дистанцию по y_top (если в get_count_men в массиве остались только люди посчитанные как вошедшие/вышедшие)
        df1.loc[:, 'distance'] = df1.last_bottom_y - df1.first_bottom_y
        df2 = df1.copy()
    # Считаем входящих (сверху вниз)
    incoming = df2.loc[df2.distance > 0].shape[0]
    # Считаем выходящих
    exiting = df2.loc[df2.distance < 0].shape[0]

    # Определяем нарушения
    V_helm = 0.8
    V_unif = 0.70
    dictinex = {'incoming': incoming, 'exiting': exiting}
    df2.loc[df2.helmet < V_helm, 'helmet'] = 0
    df2.loc[df2.helmet >= V_helm, 'helmet'] = 1
    df2.loc[df2.uniform < V_unif, 'uniform'] = 0
    df2.loc[df2.uniform >= V_unif, 'uniform'] = 1

    # соберем данные по одежде в отдельный массив для последующей проверки на этапе проверки P и R
    clothing_helmet = []
    clothing_unif = []
    for i, ds in enumerate(df2.values):
        clothing_helmet.append(int(ds[6]))
        clothing_unif.append(int(ds[7]))
    violations = df2.loc[((df2.helmet == 1) | (df2.uniform == 1)),
                         ['helmet', 'uniform', 'first_frame', 'last_frame']]   # а это сами нарушения с номерами кадров
    # Можно отдельно отобрать три группы нарушителей, но здесь все вместе
    # Пока только вывод на экран первых, можно сохранить в csv
    # return men
    return violations, incoming, exiting, clothing_helmet, clothing_unif


"""
Эта функция позволяет пройтись по всем сохраненным ранее детекциям от start_vid до end_vid 
если они существуют в папке с моделью path_model оттрековать эти боксы, наложить на соответствующие
видео и записать результат постобработки в словари
"""


def track_on_detect(path_model, tracker_path, video_source, tracker, start_vid=1, end_vid=1):
    if end_vid == 1:
        length = len([f for f in os.listdir(path_model)
                      if f.endswith('.npy') and os.path.isfile(
            os.path.join(path_model, f))])  # подсчитаем количество видео в папке
    else:
        length = end_vid

    # создаем пустые словари которые будем наполнять предсказаниями вошедших /вышедших первым  и вторым алгоритмом
    d_in1 = {str(n): 0 for n in list(range(start_vid, length + 1))}
    d_out1 = {str(n): 0 for n in list(range(start_vid, length + 1))}
    d_in2 = {str(n): 0 for n in list(range(start_vid, length + 1))}
    d_out2 = {str(n): 0 for n in list(range(start_vid, length + 1))}

    # создаем пустые словари которые будем наполнять предсказаниями нарушениями по каске (используются в следующем разделе)
    # создаем пустые словари которые будем наполнять предсказаниями нарушениями по униформе (используются в следующем разделе)
    d_helmet = {str(n): [] for n in list(range(start_vid, length + 1))}
    d_unif = {str(n): [] for n in list(range(start_vid, length + 1))}

    for N in range(start_vid, length + 1):  # устанавливаем какие видео смотрим
        try:
            with open(path_model + f'{N}.npy',
                      'rb') as files:  # Загружаем объект содержащий формат исходного изображения и детекции
                all_boxes_and_shp = np.load(files, allow_pickle=True)
                orig_shp = all_boxes_and_shp[0]  # Здесь формат
                all_boxes = all_boxes_and_shp[1]  # Здесь боксы
                out_boxes = tracking_on_detect(all_boxes, tracker,
                                               orig_shp)  # Отправляем боксы в трекинг + пробрасываем мимо трекинга каски и нетрекованные боксы людей
                create_video_with_bbox(out_boxes, video_source + f'{N}.mp4',
                                       path_model + tracker_path + f'{N}_track.mp4')  # функция отрисовки боксов на соответсвующем видео
                # out_boxes_pd = pd.DataFrame(out_boxes)
                # out_boxes_pd.to_excel(path + tracker_path + f"df_{N}_{round(orig_shp[1])}_.xlsx") # сохраняем что бы было)
                men = get_men(
                    out_boxes)  # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров где был зафиксирован айди человека + каска и жилет в его бб и без них)
                men_clean, incoming1, exiting1 = get_count_men(men, orig_shp[
                    0])  # здесь переназначаем айди входящий/выходящий (временное решение для MVP, надо думать над продом)
                violation, incoming2, exiting2, df, clothing_helmet, clothing_unif = get_count_vialotion(men_clean,
                                                                                                         orig_shp[
                                                                                                             0])  # Здесь принимаем переназначенные айди смотрим нарушения а также повторно считаем входящих по дистанции, проверяем

                d_in1[f'{N}'] = incoming1
                d_out1[f'{N}'] = exiting1
                d_in2[f'{N}'] = incoming2
                d_out2[f'{N}'] = exiting2
                d_helmet[f'{N}'] = clothing_helmet
                d_unif[f'{N}'] = clothing_unif

        except:
            print(f'данные по видео {N}: отсутствуют')
    return d_in1, d_out1, d_in2, d_out2, d_helmet, d_unif
