"""
Модуль для создания ч/б картинок

"""
import os
import shutil

from PIL import Image


def convert_colored_to_bw(path_in: str, path_out: str, col_channels: int = 1):
    """
    :param path_in:  the path to main folder with labels ('train' and 'val' do not specify)
    :param path_out: the path to a new dataset with black and white images
    :param col_channels: number of black and white image channels 1 or 3
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

    # Функция преобразования картинок
    def transformer(folder: str):
        """
        :param folder: Can take values 'train' or 'val'
        :return:

        The function goes through the folder with pictures, converts each into a black and white
        image with 1 or 3 channels and writes it to a new folder (path_out)

        """

        # Проходим по всем файлам в каталоге по указанному пути
        for filename in sorted(os.listdir(path_in + folder + img)):

            old_img = Image.open(os.path.join(path_in + folder + img,
                                              filename))
            # old_img = tf.keras.utils.load_img(os.path.join(path_in_local + folder + img,
            #                                              filename))

            if col_channels == 1:
                image_file = old_img.convert('1')  # convert image to monochrome - this works
            elif col_channels == 3:
                image_file = old_img.convert('L')  # convert image to black and white
            else:
                print('Количество каналов м.б. или 1 или 3. Укажите параметр корректно')
                return

            image_file.save(path_out + folder + img + filename)

    def copy_labels(folder: str):
        """
        :param folder: Can take values 'train' or 'val'
        :return:

        The function of copying markup. Passes through the folder with labels
        and copies them to the corresponding folders of the new dataset (path_out)
        """
        for file_name in os.listdir(path_in + folder + lbl):
            # construct full file path
            source = path_in + folder + lbl + file_name
            destination = path_out + folder + lbl + file_name

            if os.path.isfile(source):
                shutil.copy(source, destination)

    transformer(train)  # формируем ч/б train
    transformer(val)  # формируем ч/б val
    copy_labels(train)  # копируем лейблы train
    copy_labels(val)  # копируем лейблы val


def do_convert():
    """
    Пример выполнения
    Returns
    -------

    """
    convert_colored_to_bw('d:\\AI\\2023\\dataset-v1.1/',
                          'd:/AI/2023/dataset-v1.1/bw31/', 1)


# do_convert()
