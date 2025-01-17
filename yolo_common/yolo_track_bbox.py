from pathlib import Path

import cv2
import torch
from ultralytics.data.augment import LetterBox

from tools.change_bboxes import change_bbox
from configs import WEIGHTS
from tools.labeltools import is_human_class
from tools.save_txt_tools import yolo_load_detections_from_txt
from tools.track_objects import get_classes
from trackers.multi_tracker_zoo import create_tracker
from utils.torch_utils import select_device, time_synchronized


class YoloTrackBbox:

    def __init__(self, half=False, device=''):

        self.half = half
        self.device = select_device(device)

        if self.half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # print(f"device = {self.device}, half = {self.half}")

        self.reid_weights = ""

        self.letter_box = LetterBox()

    def det_to_tensor(self, df, w, h):
        res = []
        for item in df.values:
            ww = item[5] * w
            hh = item[6] * h
            x1 = item[3] * w
            y1 = item[4] * h
            conf = item[7]
            clss = item[2]
            res.append([x1, y1, ww + x1, hh + y1, conf, clss])

        return torch.tensor(res, device=self.device)

    def track(self, source, txt_source, tracker_type, tracker_config,
              reid_weights="osnet_x0_25_msmt17.pt",
              conf_threshold=0.3,
              iou=0.4,
              classes=None, change_bb=False, log: bool = True):

        file_id = Path(source).stem

        tracks_t1 = time_synchronized()

        self.reid_weights = Path(WEIGHTS) / reid_weights

        if classes is None:
            classes = get_classes()

        tracker_dict = {}

        input_video = cv2.VideoCapture(source)

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количество кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        need_camera_update = False
        need_image = False
        # создаем трекеры по каждому классу

        for class_id in classes:
            tracker_dict[class_id] = create_tracker(tracker_type, tracker_config,
                                                    self.reid_weights,
                                                    self.device, self.half, fps=fps)
            need_camera_update = hasattr(tracker_dict[class_id], 'camera_update')
            need_image = hasattr(tracker_dict[class_id], 'need_image')

            if need_image:
                need_image = tracker_dict[class_id].need_image
            else:
                need_image = True

        need_image = need_camera_update or need_image

        if log:
            print(f"read frame from video = {need_camera_update}")

        file_t1 = time_synchronized()

        df_bbox = yolo_load_detections_from_txt(txt_source)

        file_t2 = time_synchronized()

        # боксы нужно фильтровать по классам

        if classes is not None and len(classes) > 0:
            # 2 индекс класс
            mask = df_bbox[2].isin(classes)

            # bbox, которые попадают в трек
            bbox_track = df_bbox[mask]
            # не попадают
            bbox_no_track = df_bbox[~mask]
        else:
            bbox_track = df_bbox
            bbox_no_track = None

        if log:
            print(f"file '{txt_source}' read in ({(1E3 * (file_t2 - file_t1)):.1f}ms)")
            print(f"input = {source}, w = {w}, h = {h}, fps = {fps}, "
                  f"frames_in_video = {frames_in_video}")

        file_name = Path(source).name

        curr_frame, prev_frame = None, None

        results = []

        d_tracks_sum = 0
        d_df_sum = 0
        d_group_sum = 0
        d_video_sum = 0
        # пустой тензор, передается трекеру, когда нет детекций
        empty_tensor = torch.zeros(0, 6)

        for frame_id in range(frames_in_video):

            frame_id = int(frame_id)

            video_t0 = time_synchronized()

            if need_image:
                ret, frame = input_video.read()

                if frame is None:
                    print(f"frame {frame_id} is None, ret = {ret}")
                    continue
            else:
                ret, frame = True, None

            video_t1 = time_synchronized()

            curr_frame = frame

            if need_camera_update:
                if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                    for key in tracker_dict.keys():
                        tr = tracker_dict.get(key)
                        tr.camera_update(prev_frame, curr_frame)

            t0 = time_synchronized()

            df_bbox_det = bbox_track[bbox_track[0] == frame_id]

            t1 = time_synchronized()

            tracker_outputs = []

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak

                group_t0 = time_synchronized()
                # group = df_bbox_det[df_bbox_det[0] == frame_id]

                for key in tracker_dict.keys():

                    df_by_class = df_bbox_det[df_bbox_det[2] == key]

                    if len(df_by_class) > 0:
                        predict = self.det_to_tensor(df_by_class, w, h)

                        # ббокс если и меняем, то только у человека
                        if is_human_class(key):
                            predict = change_bbox(predict, change_bb, file_id)
                    else:
                        predict = empty_tensor

                    tr = tracker_dict.get(key)
                    out = tr.update(predict.cpu(), frame)
                    tracker_outputs.extend(out)

                group_t1 = time_synchronized()

                dets = 0
                empty_conf_count = 0

                track_t1 = time_synchronized()

            track_t2 = time_synchronized()
            for det_id, detection in enumerate(tracker_outputs):  # detections per image

                dets += 1

                x1 = float(detection[0]) / w
                y1 = float(detection[1]) / h
                x2 = float(detection[2]) / w
                y2 = float(detection[3]) / h

                left = min(x1, x2)
                top = min(y1, y2)
                width = abs(x1 - x2)
                height = abs(y1 - y2)

                if detection[6] is None:
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
                        float(detection[6])]

                results.append(info)

            t2 = time_synchronized()

            prev_frame = frame

            d_track = track_t2 - track_t1
            d_df = t1 - t0
            d_group = group_t1 - group_t0
            d_video = video_t1 - video_t0

            d_tracks_sum += d_track
            d_df_sum += d_df
            d_group_sum += d_group
            d_video_sum += d_video

            if log:
                detections_info = f"({dets} tracks)"
                empty_conf_count_str = \
                    f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"

                print(f'{file_name} ({frame_id + 1}/{frames_in_video}) Done.'
                      f' Track = ({(1E3 * d_track):.1f}ms), '
                      f' df = ({(1E3 * d_df):.1f}ms), ({(1E3 * (t2 - t1)):.1f}ms) tracking, '
                      f' d_group = ({(1E3 * d_group):.1f}ms), '
                      f' d_video = ({(1E3 * d_video):.1f}ms), '
                      f'{detections_info} {empty_conf_count_str}')

        input_video.release()

        tracks_t2 = time_synchronized()

        if log:
            print(f'Total tracking ({(1E3 * (tracks_t2 - tracks_t1)):.1f}ms), '
                  f'd_tracks_sum = ({(1E3 * d_tracks_sum):.1f}ms),'
                  f'd_group_sum = ({(1E3 * d_group_sum):.1f}ms),'
                  f'd_video_sum = ({(1E3 * d_video_sum):.1f}ms),'
                  f'd_df_sum = ({(1E3 * d_df_sum):.1f}ms)')

        # которых не трекали, запишем тоже, но с id -1
        if bbox_no_track is not None:
            for detection in bbox_no_track.values:
                info = [detection[0],
                        detection[3], detection[4],  # l t
                        detection[5], detection[6],  # w h
                        # id
                        -1,
                        # cls
                        int(detection[2]),
                        # conf
                        float(detection[7])]
                results.append(info)
        return results


def test():
    txt_source_folder = "D:\\AI\\2023\\Detect\\2023_03_29_10_35_01_YoloVersion.yolo_v7_detect"

    txt_source = Path(txt_source_folder) / "17.txt"

    df_bbox = yolo_load_detections_from_txt(txt_source)

    frame_id = 23

    bbox_track = df_bbox

    df_bbox_det = bbox_track[bbox_track[0] == frame_id]

    print(df_bbox_det)

    group = df_bbox_det[df_bbox_det[0] == frame_id]

    print(group)


if __name__ == '__main__':
    # test()
    pass
