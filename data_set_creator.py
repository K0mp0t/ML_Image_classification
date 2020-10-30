import h5py
import numpy as np
import os
import time
from PIL import Image
import random
import imageio
from data_augmentator import augmentate


def get_last_index(directory_path):  # used to get max index of files in directory
    files = os.listdir(directory_path)
    max_index = 0
    for file in files:
        index = int(file.split('_')[0])
        if index > max_index:
            max_index = index
    return max_index


def is_cat_image(image_path):
    return image_path.split('.')[0][-1] == '1'


def make_filenames_shift(directory_path):  # used to shift filenames after manual images deleting
    last_index = get_last_index(directory_path)
    for i in range(last_index + 1):
        if os.path.isfile(directory_path + str(i) + '_1.jpg') or os.path.isfile(directory_path + str(i) + '_0.jpg'):
            continue
        else:
            for j in range(i + 1, last_index + 1):
                if os.path.isfile(directory_path + str(j) + '_1.jpg'):
                    os.rename(directory_path + str(j) + '_1.jpg', directory_path + str(i) + '_1.jpg')
                elif os.path.isfile(directory_path + str(j) + '_0.jpg'):
                    os.rename(directory_path + str(j) + '_0.jpg', directory_path + str(i) + '_0.jpg')
                else:
                    continue
                break


def make_empty_arrays(directory_path, augmentation_coefficient):
    num_of_images = get_last_index(directory_path) + 1

    example_image = np.array(Image.open(directory_path + os.listdir(directory_path)[0])).flatten()

    num_test = num_of_images // 5
    num_train = num_of_images - num_test

    train_array_x = np.zeros((num_train*augmentation_coefficient, example_image.shape[0]))
    train_array_y = np.zeros((num_train*augmentation_coefficient, 1))
    test_array_x = np.zeros((num_test*augmentation_coefficient, example_image.shape[0]))
    test_array_y = np.zeros((num_test*augmentation_coefficient, 1))

    return train_array_x, train_array_y, test_array_x, test_array_y


def make_arrays(directory_path):  # flattens all images and puts the into two arrays (train-80%, test-20%)
    augmentation_coefficient = 1  # flip equals to *2, shift equals to *5

    train_array_x, train_array_y, test_array_x, test_array_y = make_empty_arrays(directory_path, augmentation_coefficient)
    images_list = os.listdir(directory_path)
    dogs_list = [dog for dog in images_list if not is_cat_image(dog)]
    cats_list = [cat for cat in images_list if is_cat_image(cat)]
    del images_list
    num_train = train_array_x.shape[0]//augmentation_coefficient
    num_test = test_array_x.shape[0]//augmentation_coefficient
    working_list = list(cats_list[:num_train//2] + dogs_list[:num_train//2])
    flag = True
    for i in range(num_train+num_test):
        if i == num_train:
            working_list = list(cats_list[num_train//2:] + dogs_list[num_train//2:])
            flag = False
        index = random.randint(0, len(working_list)-1)
        image = np.array(Image.open(directory_path+working_list[index], 'r'))
        augmented_images = augmentate(image, 5, flip=False, shift=False)
        if flag:
            for j in range(len(augmented_images)):
                train_array_x[i*augmentation_coefficient+j] = augmented_images[j].flatten()
                train_array_y[i*augmentation_coefficient+j] = 1 if working_list[index] in cats_list else 0
        else:
            for j in range(len(augmented_images)):
                test_array_x[(i-num_train)*augmentation_coefficient+j] = augmented_images[j].flatten()
                test_array_y[(i-num_train)*augmentation_coefficient+j] = 1 if working_list[index] in cats_list else 0
        del working_list[index]

    return train_array_x, train_array_y, test_array_x, test_array_y


def make_h5_file(h5_path, images_path):  # used to make a h5 file out of train and test set arrays
    make_filenames_shift(images_path)
    h5 = h5py.File(h5_path + 'data_big.h5', 'w')
    train_x, train_y, test_x, test_y = make_arrays(images_path)
    h5.create_dataset('train_x', data=train_x)
    h5.create_dataset('train_y', data=train_y)
    h5.create_dataset('test_x', data=test_x)
    h5.create_dataset('test_y', data=test_y)


def make_an_image_out_of_h5(h5_file_path, image_path, image_index, dataset_name):  # used to restore an image from a h5
    dataset = h5py.File(h5_file_path, 'r')
    image = dataset[dataset_name][image_index]
    restored_image = np.reshape(image, (64, 64, 3))
    imageio.imwrite(image_path, restored_image)


if __name__ == '__main__':
    start = time.time()
    # make_filenames_shift(r'E:/Peter/parsed_images/')
    make_h5_file(r'E:\Peter\\', r'E:\Peter\parsed_images\\')
    # make_an_image_out_of_h5(r'E:\Peter\data.h5', r'E:\Peter\new_image.jpg', 66, 'train_x')

    print('done in %i ms' % round((time.time() - start) * 1000))
