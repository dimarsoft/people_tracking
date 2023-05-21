import operator
from collections import defaultdict
from pathlib import Path

import numpy as np
import redis
import torch
from ultralytics import YOLO
from ultralytics.tracker.track import on_predict_start
from ultralytics.yolo.v8.detect import DetectionPredictor


class MyDetectionPredictor(DetectionPredictor):

    def __init__(self, *args, barrier=370, task_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dict_id_result = defaultdict(list)
        # self.dict_violations = defaultdict(list)
        self.data_id_violations = defaultdict(list)
        self.direction_walk = {}
        self.track_data = None
        self.barrier_y = barrier
        # self.trackers = BOTSORT()
        self.task_id = task_id
        print('send task', task_id)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    @staticmethod
    def cross_box(box1, box2):
        """box1 in box2 ? """
        mask = (box1[0] >= box2[0] and box1[2] <= box2[2]) and (
                box1[1] >= box2[1] and box1[3] <= box2[3])
        return mask

    def write_result_violations(self, tracks, results, idx):
        """Значения ключей будут категории нарушения:
            1: человек без каски и без жилета,
            2: человек с каской без жилета,
            3: человек с жилетом но без каски
        """
        list_violations = [1, 2, 3]
        name_video_batch = self.batch[4]
        if not len(results[idx].boxes):
            for id in tracks[:, 4]:
                self.data_id_violations[id].append((list_violations[0], name_video_batch))
        else:
            for track in tracks:
                list_cross = []
                for box in results[idx].boxes:
                    id = track[4]
                    result_cross = self.cross_box(box.xyxy.numpy().flatten(), track.flatten())
                    if result_cross:
                        box_cls = int(box.cls)
                        box_conf = float(box.conf)
                        # self.dict_violations[id].append((box_cls, box_conf))
                        list_cross.append(box_cls + 1)
                    else:
                        continue  # TODO
                if list_cross:
                    set_violations = (set(list_violations[1:])).intersection(set(list_cross))
                    if len(set_violations) == 1:
                        self.data_id_violations[id].append((*set_violations, name_video_batch))
                else:
                    self.data_id_violations[id].append((list_violations[0], name_video_batch))

    def write_results(self, idx, results, batch):
        log_string = ''
        det_track = results[idx].boxes.cpu().numpy()
        det_track_not_human = det_track[det_track.cls != 0]
        det_track_human = det_track[det_track.cls == 0]
        if det_track_human:
            self.results[idx].update(boxes=torch.as_tensor(det_track_not_human.boxes))
            string_result = super().write_results(idx, self.results, batch)


            # log_string += string_result#TODO  надо ли?

        im0s = self.batch[2]
        im0s = im0s if isinstance(im0s, list) else [im0s]

        if hasattr(self, 'trackers'):
            tracks = self.trackers[0].update(det_track_human, im0s[0])
            self.track_data = tracks
        tracks = self.trackers[0].update(det_track_human, im0s[0])
        self.track_data = tracks
        if len(tracks) == 0:
            log_string += super().write_results(idx, results, batch)
            self.write_redis(log_string)
            self.write_redis(self.batch[4])
            return log_string
        # central
        id_array = tracks[:, 4]
        x = ((abs((tracks[:, 1] - tracks[:, 0])) / 2) + tracks[:, 0])
        y = ((abs((tracks[:, 3] - tracks[:, 2])) / 2) + tracks[:, 2])
        # determine incoming or outgoing#TODO REF
        for id, coord_x, coord_y in zip(id_array, x, y):
            id_human = int(id)
            if not id_human in self.direction_walk:
                if coord_y - self.barrier_y < 0:
                    # in
                    self.direction_walk[id_human] = {'operator': operator.gt, 'direction': 'enter',
                                                     'status_track': False}
                else:
                    # out
                    self.direction_walk[id_human] = {'operator': operator.lt, 'direction': 'out', 'status_track': True}

            operator_walk = self.direction_walk[id_human].get('operator')

            if operator_walk(coord_y, self.barrier_y):
                if self.direction_walk[id_human]['direction'] == 'out':
                    self.direction_walk[id_human].update({'status_track': False, 'operator': operator.gt})
                elif self.direction_walk[id_human]['direction'] == 'enter':  # TODO REF
                    self.direction_walk[id_human].update({'status_track': True, 'operator': operator.lt})

            self.dict_id_result[id_human].append((int(coord_x), int(coord_y)))

        tracks_array = tracks[
            np.in1d(tracks[:, 4], [id for id in self.direction_walk if self.direction_walk[id].get('status_track')])]
        if np.any(tracks_array):
            self.write_result_violations(tracks_array, self.results, idx)

        self.results[idx].update(boxes=torch.as_tensor(tracks[:, :-1]))
        log_string += super().write_results(idx, self.results, batch)
        self.write_redis(self.batch[4])
        # self.redis_client.set(f"log:1:{idx}", log_string)
        return log_string

    def write_redis(self, log_string, parse=True):
        if parse:
            log_string = ' '.join(log_string.split()[:-1])
        self.redis_client.set(f"task_{self.task_id}", log_string)



def init_model(track=True, model="yolov8n.pt", task_id=None):
    my_model = YOLO(model=model)
    if track:
        my_model.predictor = MyDetectionPredictor(task_id=task_id)
        my_model.predictor.save_dir = Path(__file__).resolve().parent.parent#TODO
    return my_model


def predict_model(source='1(21).mp4', task_id=None, **kwargs):
    event = 'on_predict_start'
    my_model = init_model(task_id=task_id)
    # kwargs = {}
    # conf = 0.1
    # kwargs['conf'] = conf
    kwargs['mode'] = 'track'
    my_model.add_callback(event, on_predict_start)
    my_model.predict(source=source, **kwargs)
    dict_out_in = defaultdict(int)
    for dt in my_model.predictor.direction_walk.values():
        if dt.get('direction') == 'enter':
            dict_out_in['enter'] += 1
        else:
            dict_out_in['out'] += 1

    return dict_out_in, my_model.predictor.data_id_violations


def predict_yolo(source=None):
    predict_model(source=source, save=True, classes=0, line_thickness=1, show=True)



