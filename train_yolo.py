# Мы используем веб-сервис ClearML для записи логов и графиков обучения
# Для этого необходима регистрация на сервисе
CLEARML_WEB_HOST="https://app.clear.ml"
CLEARML_API_HOST="https://api.clear.ml"
CLEARML_FILES_HOST="https://files.clear.ml"
CLEARML_API_ACCESS_KEY="please_enter_your_API_access_key" # Здесь необходимо вставить свой ключ доступа к веб-сервису ClearML
CLEARML_API_SECRET_KEY="please_enter_your_API_secret_key" # Здесь необходимо вставить свой ключ доступа к веб-сервису ClearML

from ultralytics import YOLO
from clearml import Task

task = Task.init(project_name='YOLOv8', task_name='YOLOv8s_200epochs_defimgsz')
model = YOLO('yolov8s.pt')  # Создаем объект модели, указываем путь к весам во входном параметре (веса модели предобученной на датасете COCO и предоставляемые ultralytics)
model.train(data='/hardhats_vests/dataset-v1.1/data_custom.yaml', epochs=200)  # Обучаем модель на нашем датасете
task.close()

# В результате обучение остановилось на эпохе 152, так как в течение 50 эпох метрики не улучшались
# Сохранены веса модели best.pt полученные на 102й эпохе обучения.
