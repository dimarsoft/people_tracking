import argparse
import gc
import json
from pathlib import Path

from configs import parse_yolo_version, YoloVersion
from post_processing.timur import get_camera
from tools.labeltools import TrackWorker
from tools.path_tools import get_video_files, create_session_folder
from tools.save_txt_tools import yolo7_save_tracks_to_txt, yolo7_save_tracks_to_json
from tools.video_turniket import create_turniket_video
from utils.general import set_logging
from utils.torch_utils import time_synchronized
from yolo_common.yolo_detect import create_yolo_model
from yolo_common.yolov7 import YOLO7
from yolo_common.yolov8 import YOLO8
from yolo_common.yolov8_ultralitics import YOLO8UL


def detect_single_video_yolo(yolo_version, model, source, output_folder, classes=None,
                             conf=0.3, iou=0.45, save_txt=True,
                             save_vid=False, max_frames=-1):
    print(f"start detect_single_video_yolo: {yolo_version}, source = {source}")

    source_path = Path(source)

    camera_num, w, h, fps = get_camera(source)

    model = create_yolo_model(yolo_version, model, w=w, h=h)

    detections = model.detect(
        source=source,
        conf_threshold=conf,
        iou=iou,
        classes=classes,
        max_det=300,
        max_frames=max_frames
    )

    if save_txt:
        text_path = Path(output_folder) / f"{source_path.stem}.txt"

        print(f"save detections to: {text_path}")

        yolo7_save_tracks_to_txt(results=detections, txt_path=text_path, conf=conf)

        json_file = Path(output_folder) / f"{source_path.stem}.json"

        print(f"save detections to: {json_file}")

        yolo7_save_tracks_to_json(results=detections, json_file=json_file, conf=conf)

    if save_vid:
        labels = TrackWorker.convert_tracks_to_list(detections)
        t1 = time_synchronized()
        create_turniket_video(labels, source, output_folder, max_frames=max_frames)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")

    del detections

    gc.collect()


def run_detect_yolo(yolo_info, model: str, source: str, output_folder,
                    files=None, classes=None, conf=0.3, iou=0.45,
                    save_txt=True, save_vid=False):
    """

    Args:
        iou:
        yolo_info: версия Yolo: 7 ил 8
        save_txt: сохранять бб в файл
        files: если указана папка, но можно указать имена фай1лов,
                которые будут обрабатываться. ['1', '2' ...]
        classes: список классов, None все, [0, 1, 2....]
        conf: conf
        save_vid: Создаем видео c bb
        output_folder: путь к папке для результатов работы, txt
        source: путь к видео, если папка, то для каждого видео файла запустит
        model (str): модель для YOLO8
    """

    set_logging()

    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    session_folder = create_session_folder(yolo_version, output_folder, "detect_turniket")

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['conf'] = conf
    session_info['iou'] = iou
    session_info['save_vid'] = save_vid
    session_info['files'] = files
    session_info['classes'] = classes
    session_info['save_txt'] = save_txt
    session_info['yolo_version'] = str(yolo_version)

    session_info_path = str(Path(session_folder) / 'session_info.json')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    # список файлов с видео для обработки
    list_of_videos = get_video_files(source, files)
    total_videos = len(list_of_videos)

    for i, item in enumerate(list_of_videos):
        print(f"process file: {i + 1}/{total_videos} {item}")

        detect_single_video_yolo(yolo_version, model, str(item), session_folder,
                                 classes=classes, conf=conf, iou=iou,
                                 save_txt=save_txt, save_vid=save_vid, max_frames=10)


def run_example():
    video_source = "d:\\AI\\2023\\dataset-v1.1\\test\\"
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    files = ["81", '1', '29', "20", '32']
    # files = ['20', "79", "81"]
    files = None

    model = "C:\\AI\\турникет\\TrainedModels\\" \
            "Yolo8n_turniket_batch64_epoch57_single_class_false_best.pt"

    model = "C:\\AI\\турникет\\TrainedModels\\" \
            "2023_05_13_Yolo8n_turniket_batch64_epoch57_single_class_true_best.pt"

    model = "C:\\AI\\турникет\\TrainedModels\\" \
            "2023_05_14_13_43_05_yolo8_train_yolov8x.pt_epochs_100_batch_16_single_cls_True_best.pt"




    model = "C:\\AI\\турникет\\TrainedModels\\" \
            "2023_05_15_Yolo8n_turniket_batch64_epoch201_single_class_true_best.pt"
    model = "C:\\AI\\турникет\\TrainedModels\\" \
            "2023_05_14_Yolo8s_turniket_batch32_epoch241_single_class_true_best.pt"

    model = "C:\\AI\\турникет\\TrainedModels\\" \
            "2023_05_12_17_52_39_yolo8_train_yolov8n.pt_epochs_30_batch_8_single_cls_best.pt"

    run_detect_yolo("8ul", model, video_source, output_folder, files=files, conf=0.25, save_txt=True, save_vid=True)

    pass


# запуск из командной строки: python yolo_detect.py  --yolo 7 --weights "" source ""
def run_cli(opt_info):
    yolo, source, weights, output_folder, files, save_txt, save_vid, conf, classes = \
        opt_info.yolo, opt_info.source, opt_info.weights, opt_info.output_folder, \
        opt_info.files, opt_info.save_txt, opt_info.save_vid, opt_info.conf, opt_info.classes

    run_detect_yolo(yolo, weights, source, output_folder,
                    files=files, conf=conf, iou=opt_info.iou,
                    save_txt=save_txt, save_vid=save_vid, classes=classes)


if __name__ == '__main__':
    # lst = range(44, 72)
    # lst = [f"{x}" for x in lst]
    # print(lst)

    run_example()

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', type=int, help='7, 8, 8ul')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--files', type=str, default=None, help='files names list')  # files from list
    parser.add_argument('--output_folder', type=str, help='output_folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_vid', action='store_true', help='save results to *.mp4')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # print(opt)

    # run_cli(opt)
