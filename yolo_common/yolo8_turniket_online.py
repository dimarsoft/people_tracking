from pathlib import Path
from typing import Union

import cv2
import torch
from ultralytics import YOLO

from tools.labeltools import DetectedTrackLabel, Labels
from tools.video_tools import VideoInfo
from tools.video_turniket import draw_on_frame
from utils.general import check_img_size, xyxy2xywh
from utils.torch_utils import select_device, time_synchronized


class YOLO8ULTurniketOnline:

    def to_label(self, dets, frame_id) -> DetectedTrackLabel:
        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        classes = clss.numpy()
        xywhs = xyxy2xywh(xyxys.numpy())
        confs = confs.numpy()

        xywhs = xywhs[0]

        return DetectedTrackLabel(Labels.human,
                                  xywhs[0], xywhs[1], xywhs[2], xywhs[3], -1, frame_id, confs[0])

    def __init__(self,
                 weights_path,
                 save_images_folder: Union[str, Path],
                 fps: int,
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

        self.conf_threshold = 0.3
        self.iou = 0.4
        self.results = []
        self.log = True
        self.w = self.imgsz[0]
        self.h = self.imgsz[1]

        self.has_track_lat_time = None
        self.start_frame_id = None
        self.end_frame_id = None

        self.video_info: VideoInfo = VideoInfo(self.w, self.h, fps)

        self.save_images_folder = Path(save_images_folder)

    def track_frame(self, frame, frame_id):

        # Inference
        t1 = time_synchronized()
        s = ""

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            predict = self.model.predict(frame,
                                         conf=self.conf_threshold,
                                         iou=self.iou,
                                         classes=None,
                                         max_det=1,
                                         imgsz=self.imgsz,
                                         stream=False)[0].boxes.data
        t2 = time_synchronized()
        predict_track = predict
        dets = 0
        if len(predict_track) > 0:
            dets += 1

            predict = predict.cpu()

            label = self.to_label(predict, frame_id)

            caption = f"{label.conf:.2f}"
            image_path = self.save_images_folder / f"{int(frame_id)}_{caption}.jpg"

            draw_on_frame(frame, 1, 1, label)

            cv2.imwrite(str(image_path), frame)

        if self.log:
            detections_info = f"{'' if dets > 0 else ', (no detections)'}"
            print(f'frame ({frame_id}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, {detections_info}')
