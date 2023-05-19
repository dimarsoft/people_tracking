import cv2
import torch
from ultralytics import YOLO
from pathlib import Path

from tools.change_bboxes import change_bbox
from configs import WEIGHTS
from tools.save_txt_tools import convert_toy7
from trackers.multi_tracker_zoo import create_tracker
from utils.datasets import LoadImages
from utils.general import check_img_size
from utils.torch_utils import select_device, time_synchronized


# from PIL import Image

class YOLO8UL:
    def __init__(self, weights_path, half=False, device='', imgsz=(640, 640)):
        self.device = select_device(device)

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")

        self.model = YOLO(weights_path)

        self.stride = 32

        self.imgsz = (check_img_size(imgsz[0], self.stride), check_img_size(imgsz[1], self.stride))

        self.names = self.model.names

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

    def detect(self, source, conf_threshold=0.3, iou=0.4, classes=None,
               max_det: int = 300,
               max_frames: int = -1,
               bw_image: bool = False):

        if bw_image:
            return self.detect_bw(source, conf_threshold, iou, classes, max_det, max_frames)

        detections = self.model.predict(source, conf=conf_threshold, iou=iou,
                                        classes=classes, imgsz=self.imgsz,
                                        stream=True, max_det=max_det)

        detections = convert_toy7(detections, save_none_id=True, max_frames=max_frames)

        return detections

    def detect_bw(self, source, conf_threshold=0.3, iou=0.4, classes=None,
                  max_det: int = 300,
                  max_frames: int = -1):

        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        detections = []

        index = 0

        for path, img, im0s, vid_cap in dataset:

            if 0 < max_frames < index:
                break
            index += 1

            # пока костыль, но нужно переделать
            img0_gray = cv2.cvtColor(im0s, cv2.COLOR_BGR2GRAY)

            image_path = f"{index}.jpg"
            cv2.imwrite(str(image_path), img0_gray)
            img0 = cv2.imread(image_path)

            detection = self.model.predict(img0, conf=conf_threshold, iou=iou,
                                           classes=classes, imgsz=self.imgsz,
                                           stream=False, max_det=max_det)
            detections.extend(detection)

        detections = convert_toy7(detections, save_none_id=True, max_frames=max_frames)

        return detections

    def track(self, source, tracker_type, tracker_config, reid_weights="osnet_x0_25_msmt17.pt",
              conf_threshold=0.3, iou=0.4,
              classes=None, change_bb=False, log: bool = True) -> list:

        self.reid_weights = Path(WEIGHTS) / reid_weights

        input_video = cv2.VideoCapture(source)

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количество кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        tracker = create_tracker(tracker_type, tracker_config, self.reid_weights,
                                 self.device, self.half, fps=fps)

        update_camera = hasattr(tracker, 'camera_update')

        if log:
            print(f"input = {source}, w = {w}, h = {h}, fps = {fps}, "
                  f"frames_in_video = {frames_in_video}")

        curr_frame, prev_frame = None, None

        results = []

        for frame_id in range(frames_in_video):
            ret, frame = input_video.read()

            if not ret:
                continue

            # Inference
            t1 = time_synchronized()
            s = ""

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                predict = self.model.predict(frame,
                                             conf=conf_threshold,
                                             iou=iou,
                                             classes=classes,
                                             imgsz=self.imgsz,
                                             stream=False)[0].boxes.data

                t2 = time_synchronized()

                curr_frame = frame

                if update_camera:
                    # camera motion compensation
                    if prev_frame is not None and curr_frame is not None:
                        tracker.camera_update(prev_frame, curr_frame)

                dets = 0
                empty_conf_count = 0
                predict_track = predict
                if len(predict_track) > 0:

                    predict_track = change_bbox(predict_track, change_bb, clone=True)

                    dets += 1

                    if log:
                        # Print results
                        for c in predict_track[:, 5].unique():
                            n = (predict_track[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    tracker_outputs = tracker.update(predict_track.cpu(), frame)

                    # Process detections [f, x1, y1, x2, y2, track_id, class_id, conf]

                    for det_id, detection in enumerate(tracker_outputs):  # detections per image

                        x1 = float(detection[0]) / w
                        y1 = float(detection[1]) / h
                        x2 = float(detection[2]) / w
                        y2 = float(detection[3]) / h

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
                        results.append(info)

                t3 = time_synchronized()

                prev_frame = frame

                if log:
                    detections_info = f"{s}{'' if dets > 0 else ', (no detections)'}"
                    empty_conf_count_str = \
                        f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"
                    # Print total time (preprocessing + inference + NMS + tracking)

                    # Print time (inference + NMS)
                    print(f'frame ({frame_id + 1}/{frames_in_video}) Done. '
                          f'({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                          f'({(1E3 * (t3 - t2)):.1f}ms) track, '
                          f'{detections_info} {empty_conf_count_str}')

        input_video.release()

        return results

    def train(self, **kwargs):
        self.model.train(**kwargs)
