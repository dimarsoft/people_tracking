import os
import shutil
from pathlib import Path
from typing import Union


def delete_and_change(src_labels: Union[str, Path], out_folder: Union[str, Path]) -> None:
    src_labels = Path(src_labels)
    out_folder = Path(out_folder)

    count_0 = 0
    count_1 = 0

    total_count = 0

    for label in src_labels.iterdir():
        if label.is_file() and label.suffix == ".txt":
            # пустые (нет детекций), но размер не всегда ноль
            if label.stat().st_size > 1:

                out_l = out_folder / label.name

                with open(out_l, 'a') as out_file:
                    with open(label, 'r') as src_file:
                        for line in src_file:
                            total_count += 1
                            words = line.split(' ')
                            if words[0] == '0':
                                new_line = " ".join(words)
                                out_file.write(new_line)

                                count_0 += 1

                            if words[0] == '3':
                                words[0] = '1'
                                new_line = " ".join(words)
                                out_file.write(new_line)

                                count_1 += 1
            else:
                shutil.copy(label, out_folder)

    print(f"count_0 = {count_0}, count_1 = {count_1}, total_count = {total_count}")


def copy_labels_image(src_labels: Union[str, Path], src_images: Union[str, Path],
                      out_folder: Union[str, Path], lb_info_file: Union[str, Path]) -> None:
    """
    Туулза для датасета с двумя классами.
    Labels разделили по 500 файлов, теперь нужно и картинки также разбить.
    И добавить labels.txt
    Parameters
    ----------
    src_labels
        Папка где находится пачка с txt.
    src_images
        Картинки.
    out_folder
        Папка, в которую запишется результат.
        В ней будет:
            папка labels c не пустыми файлами txt + labels.txt
            папка images c картинками.
    lb_info_file
        labels.txt.

    Returns
    -------

    """
    src_labels = Path(src_labels)
    src_images = Path(src_images)

    out_folder = Path(out_folder)
    labels_folder = out_folder / f"labels"
    images_folder = out_folder / f"images"

    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    shutil.copy(lb_info_file, labels_folder)

    for label in src_labels.iterdir():
        if label.is_file() and label.suffix == ".txt":
            # пустые (нет детекций), но размер не всегда ноль
            if label.stat().st_size > 1:
                shutil.copy(label, labels_folder)
                image_file = src_images / f"{label.stem}.jpg"

                shutil.copy(image_file, images_folder)

    pass


def change_labels():
    source_folder = Path('c:\\AI\\02.05.2023\\train\\labels\\')
    out_folder = Path('c:\\AI\\test\\train')

    delete_and_change(source_folder, out_folder)

    source_folder = Path('c:\\AI\\02.05.2023\\val\\labels\\')
    out_folder = Path('c:\\AI\\test\\val')

    delete_and_change(source_folder, out_folder)


def split_labels():
    train_batches = 8
    val_batches = 2

    source_folder = Path('D:\\AI\\2023\\02.05.2023')

    out_folder = Path('D:\\AI\\2023\\02.05.2023\\ds')

    train_images = "D:\\AI\\2023\\dataset-v1.1\\train\\images"
    val_images = "D:\\AI\\2023\\dataset-v1.1\\val\\images"

    lb_info_file = source_folder / "labels.txt"

    for i in range(train_batches):
        lb = source_folder / "train" / f"labels_new{i + 1}"

        batch_out_folder = out_folder / f"train_batch_{i + 1}"

        copy_labels_image(lb, train_images, batch_out_folder, lb_info_file)

    for i in range(val_batches):
        lb = source_folder / "val" / f"labels_new{i + 1}"

        batch_out_folder = out_folder / f"val_batch_{i + 1}"

        copy_labels_image(lb, val_images, batch_out_folder, lb_info_file)


if __name__ == '__main__':
    split_labels()
    # change_labels()
