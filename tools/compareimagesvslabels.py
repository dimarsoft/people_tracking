"""
Модуль для проверки соответствия файлов картинок и файлов разметки

"""

import os


def image_labels_compare(path: str):
    """
    :param path:  The path to the folder containing the dataset to check
    :return:

    The function to check the correspondence of pictures and labels
    of the dataset in folders 'train' and 'val'
    """

    train = 'train/'
    val = 'val/'
    img = 'images/'
    lbl = 'labels/'

    def compare(folder: str, key: int = 0):
        """
        :param folder: Can take values 'train' or 'val'
        :param key: 0 or 1 (0 - go through the images, search for missing labels,
                            1 - on the contrary, the passage through the labels )
        :return:

        The function of checking the correspondence of images and markup.
        Works both ways depending on the parameters
        """
        if key == 0:
            current = img
            target = lbl
            extension = '.txt'
        else:
            current = lbl
            target = img
            extension = '.jpg'

        for filename in sorted(os.listdir(path + folder + current)):
            img_name = os.path.splitext(filename)[0] + extension
            for _, _, files in os.walk(path + folder + target):

                if img_name not in files:
                    print(f'Файл {img_name} отсутствует для метки {folder}{filename}')

    print('train проверка отсутствующих картинок')
    compare(train)  # копируем лейблы и картинки train
    print('train проверка отсутствующих дублов')
    compare(val)  # копируем лейблы и картинки val
    print('val проверка отсутствующих картинок')
    compare(train, 1)  # копируем лейблы и картинки train
    print('val проверка отсутствующих дублов')
    compare(val, 1)  # копируем лейблы и картинки val
