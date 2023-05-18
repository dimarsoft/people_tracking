"""
Модуль для поиска файлов картинок по файлу разметки

"""

import os
import shutil


def image_search_by_label(path_in: str, path_in_img: str, path_out: str):
    """
    :param path_in: the path to main folder with labels ('train' and 'val' do not specify)
    :param path_in_img: the path to main folder with images ('train' and 'val' do not specify)
    :param path_out: the path to a new dataset filled with images and their corresponding labels
    :return:

    At the beginning, folders are created with an existence check.
    If the main folder exists, sub folders are NOT created.
    It is assumed that the code has already worked earlier.
    or the folders are created manually.
    """

    train = 'train/'
    val = 'val/'
    img = 'images/'
    lbl = 'labels/'

    if not os.path.exists(path_out):
        os.mkdir(path_out)
        os.mkdir(path_out + train)
        os.mkdir(path_out + val)
        os.mkdir(path_out + train + img)
        os.mkdir(path_out + train + lbl)
        os.mkdir(path_out + val + img)
        os.mkdir(path_out + val + lbl)

    # Функция преобразования имени и поиска картинок
    def image_search(folder: str):
        """

        :param folder: responsible for folders 'val' or 'train'
        :return:

        We go through all the labels in the folder along the specified path.
        If found, copy the label and copy the picture to the new output folder.
        """

        for filename in sorted(os.listdir(path_in + folder + lbl)):
            img_name = os.path.splitext(filename)[0] + '.jpg'
            for _, _, files in os.walk(path_in_img + folder + img):

                if img_name in files:
                    shutil.copy(path_in + folder + lbl + filename, path_out + folder + lbl +
                                filename)
                    shutil.copy(path_in_img + folder + img + img_name, path_out + folder + img +
                                img_name)
                else:
                    print(f'Файл {img_name} отсутствует для метки {folder}{filename}')

    # копируем лейблы и картинки train
    image_search(train)

    # копируем лейблы и картинки val
    image_search(val)
