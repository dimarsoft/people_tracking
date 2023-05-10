from time import time

import torch
from ultralytics import YOLO
from pathlib import Path

from configs import WEIGHTS
from tools.exception_tools import print_exception
from tools.video_tools import VideoInfo
from trackers.multi_tracker_zoo import create_tracker
from utils.general import check_img_size
from utils.torch_utils import select_device, time_synchronized


def time_synch():
    return time_synchronized()


def elapsed_sec(t: time):
    return (time_synch() - t) * 100


class SaveResultsBase:
    def update(self, results: list, start_id, end_id, video_info: VideoInfo):
        pass


class PrintSaveResults(SaveResultsBase):
    def update(self, results: list, start_id, end_id, video_info: VideoInfo):
        print(f"start_id = {start_id}, end_id = {end_id}, results = {len(results)}")


class YOLO8ULOnline:
    def __init__(self,
                 weights_path,
                 tracker_type,
                 tracker_config,
                 fps: int,
                 saver: SaveResultsBase,
                 half=False,
                 device='',
                 imgsz=(640, 640)):
        self.device = select_device(device)

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")

        self.model = YOLO(weights_path)

        self.imgsz = (check_img_size(imgsz[0]), check_img_size(imgsz[1]))

        print(f"imgsz = {self.imgsz}")
        self.names = self.model.names

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

        self.tracker = create_tracker(tracker_type, tracker_config, self.reid_weights, self.device, self.half, fps=fps)

        self.conf_threshold = 0.3
        self.iou = 0.4
        self.results = []
        self.log = True
        self.w = self.imgsz[0]
        self.h = self.imgsz[1]

        self.has_track_lat_time = None
        self.start_frame_id = None
        self.end_frame_id = None

        self.saver = saver
        self.video_info: VideoInfo = VideoInfo(self.w, self.h, fps)

    def reset(self):
        self.has_track_lat_time = None
        self.start_frame_id = None
        self.end_frame_id = None

    def finish_results(self):
        try:
            self.saver.update(self.results, self.start_frame_id, self.end_frame_id, self.video_info)
        except Exception as ex:
            print_exception(ex, "finish_results")

        self.reset()
        self.results = []

    def track_frame(self, frame, frame_id):

        # Inference
        t1 = time_synchronized()
        s = ""

        if self.start_frame_id is None:
            self.start_frame_id = frame_id

        self.end_frame_id = frame_id

        has_track = False

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            predict = self.model.predict(frame,
                                         conf=self.conf_threshold,
                                         iou=self.iou,
                                         classes=None,
                                         imgsz=self.imgsz,
                                         stream=False)[0].boxes.data

            t2 = time_synchronized()

            dets = 0
            empty_conf_count = 0
            predict_track = predict
            if len(predict_track) > 0:

                # predict_track = change_bbox(predict_track, change_bb, clone=True)

                dets += 1

                if self.log:
                    # Print results
                    for c in predict_track[:, 5].unique():
                        n = (predict_track[:, 5] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                tracker_outputs = self.tracker.update(predict_track.cpu(), frame)

                has_track = len(tracker_outputs) > 0

                for det_id, detection in enumerate(tracker_outputs):  # detections per image

                    x1 = float(detection[0]) / self.w
                    y1 = float(detection[1]) / self.h
                    x2 = float(detection[2]) / self.w
                    y2 = float(detection[3]) / self.h

                    left = min(x1, x2)
                    top = min(y1, y2)
                    width = abs(x1 - x2)
                    height = abs(y1 - y2)

                    track_conf = detection[6]

                    if track_conf is None:
                        # print("detection[6] is None")
                        empty_conf_count += 1
                        continue

                    info = [frame_id,
                            left, top,
                            width, height,
                            # id
                            int(detection[4]),
                            # cls
                            int(detection[5]),
                            # conf
                            float(track_conf)]

                    # print(info)
                    self.results.append(info)

            t3 = time_synchronized()

            if self.log:
                detections_info = f"{s}{'' if dets > 0 else ', (no detections)'}"
                empty_conf_count_str = f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"
                print(f'frame ({frame_id}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                      f'({(1E3 * (t3 - t2)):.1f}ms) track, '
                      f'{detections_info} {empty_conf_count_str}')

        if has_track:
            self.has_track_lat_time = time_synch()
        else:
            if self.has_track_lat_time is not None:
                if elapsed_sec(self.has_track_lat_time) > 0:
                    self.finish_results()

