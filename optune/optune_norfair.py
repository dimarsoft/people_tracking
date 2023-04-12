import optuna
from optuna.study import StudyDirection

from configs import get_detections_path
from optune.optune_tools import save_result, reset_seed
from yolo_optune import run_track_yolo


def objective_norfair(trial):
    """
  ecc: true
  max_age: 30
  max_cosine_distance: 0.1594374041012136
  max_iou_distance: 0.5431835667667874
  nms_max_overlap: 1.0
  n_init: 3
  nn_budget: 100
  embedder: "mobilenet"
  # embedder - "mobilenet",  "torchreid",  "clip_RN50",  "clip_RN101",  "clip_RN50x4",
  #            "clip_RN50x16",  "clip_ViT-B/32",  "clip_ViT-B/16",

    """

    reset_seed()

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"

    files = None
    # files = ['1', "2", "3"]
    # files = ["6"]

    # classes = [0]
    classes = None

    change_bb = None  # pavel_change_bbox  # change_bbox

    test_func = "timur"

    txt_source_folder = str(get_detections_path())

    """

    ecc = trial.suggest_categorical('ecc', [True, False])
    max_age = (trial.suggest_int('max_age', 1, 10))
    max_cosine_distance = trial.suggest_float('max_cosine_distance', 0.1, 0.2, log=True)
    max_iou_distance = trial.suggest_float('max_iou_distance', 0.3, 0.6, log=True)
    nms_max_overlap = trial.suggest_float('nms_max_overlap', 1, 3, log=True)
    n_init = trial.suggest_int('n_init', 2, 6, step=2)
    nn_budget = trial.suggest_int('nn_budget', 80, 120, step=10)
    """

    distance_function = trial.suggest_categorical('DISTANCE_FUNCTION',
                                                  ["frobenius",  "mean_manhattan", "mean_euclidean",
                                                   "iou", "iou_opt"])

    # "frobenius" # mean_manhattan, mean_euclidean, iou, iou_opt
    # test_func = trial.suggest_categorical('test_func', ["timur", "group_3"])
    tracker_name = "norfair"

    tracker_config = \
        {
            "NORFAIR_TRACK":
                {
                    "DISTANCE_FUNCTION": distance_function,  # mean_manhattan, mean_euclidean, iou, iou_opt
                    "DISTANCE_THRESHOLD": 500,
                    "HIT_COUNTER_MAX": 15,
                    "INITIALIZATION_DELAY": None,
                    "POINTWISE_HIT_COUNTER_MAX": 4,
                    "DETECTION_THRESHOLD": 0,
                    "PAST_DETECTIONS_LENGTH": 4,
                    "REID_DISTANCE_THRESHOLD": 0,
                    "REID_HIT_COUNTER_MAX": None
                }
        }

    cmp_results = run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                                 test_func=test_func,
                                 files=files, change_bb=change_bb, classes=classes)

    accuracy = cmp_results["total_equal_percent"]

    return accuracy


def run_optuna():
    study = optuna.create_study(direction=StudyDirection.MAXIMIZE)
    study.optimize(objective_norfair, n_trials=60, show_progress_bar=True)

    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print(f"Best hyper parameters: {trial.params}")

    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    save_result(trial, output_folder, "fastdeepsort")


if __name__ == '__main__':
    run_optuna()