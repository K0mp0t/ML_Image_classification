import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def augmentate(original_image, shift_value, flip=True, shift=True):
    """'
    :param original_image: a numpy array of shape (HEIGHT, WIDTH, 3)
    :param shift_value: number of pixels to shift image (shifting makes black bar on one side of the image)
    :param flip: boolean, which enables/disables flipping
    :param shift: boolean, which enables/disables shifting
    :return: list of augmented images
    """
    augmentated_images = [original_image]
    if flip:
        flipped_image = np.fliplr(original_image)
        augmentated_images.append(flipped_image)
    if shift:
        shifted_left = shift_left(original_image, shift_value)
        shifted_right = shift_right(original_image, shift_value)
        shifted_down = shift_down(original_image, shift_value)
        shifted_up = shift_up(original_image, shift_value)
        augmentated_images.append(shifted_left)
        augmentated_images.append(shifted_right)
        augmentated_images.append(shifted_down)
        augmentated_images.append(shifted_up)
        if flip:
            augmentated_images.append(np.fliplr(shifted_left))
            augmentated_images.append(np.fliplr(shifted_right))
            augmentated_images.append(np.fliplr(shifted_down))
            augmentated_images.append(np.fliplr(shifted_up))

    return augmentated_images


def shift_left(original_image, shift_value):
    """
    :param original_image: numpy array containing original image
    :param shift_value: number of pixels to shift image (shifting makes black bar on one side of the image)
    :return: shifted by shift value pixels image
    """
    height = original_image.shape[0]
    width = original_image.shape[1]
    shifted_image = np.array(original_image, copy=True)
    for i in range(0, height):
        for j in reversed(range(0, width)):
            if j > shift_value:
                shifted_image[i][j] = original_image[i][j - shift_value]
            else:
                shifted_image[i][j] = [255, 255, 255]  # it's RGB, so you could choose the value of filled pixels

    return shifted_image


def shift_right(original_image, shift_value):
    """
    :param original_image: numpy array containing original image
    :param shift_value: number of pixels to shift image (shifting makes black bar on one side of the image)
    :return: shifted by shift value pixels image
    """
    height = original_image.shape[0]
    width = original_image.shape[1]
    shifted_image = np.array(original_image, copy=True)
    for i in range(0, height):
        for j in reversed(range(0, width)):
            if j + shift_value < width:
                shifted_image[i][j] = original_image[i][j + shift_value]
            else:
                shifted_image[i][j] = [255, 255, 255]

    return shifted_image


def shift_down(original_image, shift_value):
    """
    :param original_image: numpy array containing original image
    :param shift_value: number of pixels to shift image (shifting makes black bar on one side of the image)
    :return: shifted by shift value pixels image
    """
    height = original_image.shape[0]
    width = original_image.shape[1]
    shifted_image = np.array(original_image, copy=True)
    for i in range(0, height):
        for j in range(0, width):
            if i > shift_value:
                shifted_image[i][j] = original_image[i - shift_value][j]
            else:
                shifted_image[i][j] = [255, 255, 255]

    return shifted_image


def shift_up(original_image, shift_value):
    """
    :param original_image: numpy array containing original image
    :param shift_value: number of pixels to shift image (shifting makes black bar on one side of the image)
    :return: shifted by shift value pixels image
    """
    height = original_image.shape[0]
    width = original_image.shape[1]
    shifted_image = np.array(original_image, copy=True)
    for i in range(0, height):
        for j in range(0, width):
            if i + shift_value < height:
                shifted_image[i][j] = original_image[i + shift_value][j]
            else:
                shifted_image[i][j] = [255, 255, 255]

    return shifted_image


if __name__ == '__main__':
    # lil tests
    image = Image.open(r'E:/Peter/parsed_images/100_1.jpg', 'r')
    image = np.array(image)
    images = augmentate(image, shift_value=10, flip=True, shift=True)
    for i in images:
        plt.imshow(i)
        plt.show()
