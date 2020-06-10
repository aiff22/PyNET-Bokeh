# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from __future__ import print_function
from scipy import misc
from PIL import Image
import imageio
import os
import numpy as np


def load_test_data(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE):

    test_directory_orig = dataset_dir + 'test/original/'
    test_directory_orig_depth = dataset_dir + 'test/original_depth/'
    test_directory_blur = dataset_dir + 'test/bokeh/'

    #NUM_TEST_IMAGES = 200
    NUM_TEST_IMAGES = len([name for name in os.listdir(test_directory_orig)
                           if os.path.isfile(os.path.join(test_directory_orig, name))])

    test_data = np.zeros((NUM_TEST_IMAGES, PATCH_HEIGHT, PATCH_WIDTH, 4))
    test_answ = np.zeros((NUM_TEST_IMAGES, int(PATCH_HEIGHT * DSLR_SCALE), int(PATCH_WIDTH * DSLR_SCALE), 3))

    for i in range(0, NUM_TEST_IMAGES):

        I = misc.imread(test_directory_orig + str(i) + '.jpg')
        I_depth = misc.imread(test_directory_orig_depth + str(i) + '.png')

        # Downscaling the image by a factor of 2
        I = misc.imresize(I, 0.5, interp='bicubic')

        # Making sure that its width is multiple of 32
        new_width = int(I.shape[1]/32) * 32
        I = I[:, 0:new_width, :]

        # Stacking the image together with its depth map
        I_temp = np.zeros((I.shape[0], I.shape[1], 4))
        I_temp[:, :, 0:3] = I
        I_temp[:, :, 3] = I_depth
        I = I_temp

        h, w, d = I.shape
        y = np.random.randint(0, w - 512)

        # Extracting random patch of width PATCH_WIDTH
        I = np.float32(I[:, y:y + PATCH_WIDTH, :]) / 255.0
        test_data[i, :] = I

        I = misc.imread(test_directory_blur + str(i) + '.jpg')
        I = np.float32(misc.imresize(I[:, y*2:y*2 + 1024, :], DSLR_SCALE / 2, interp='bicubic')) / 255.0
        test_answ[i, :] = I

    return test_data, test_answ


def load_training_batch(dataset_dir, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE, train_size):

    test_directory_orig = dataset_dir + 'train/original/'
    test_directory_orig_depth = dataset_dir + 'train/original_depth/'
    test_directory_blur = dataset_dir + 'train/bokeh/'

    # NUM_TRAINING_IMAGES = 4894
    NUM_TRAINING_IMAGES = len([name for name in os.listdir(test_directory_orig)
                           if os.path.isfile(os.path.join(test_directory_orig, name))])

    TRAIN_IMAGES = np.random.choice(np.arange(0, NUM_TRAINING_IMAGES), train_size, replace=False)

    test_data = np.zeros((train_size, PATCH_HEIGHT, PATCH_WIDTH, 4))
    test_answ = np.zeros((train_size, int(PATCH_HEIGHT * DSLR_SCALE), int(PATCH_WIDTH * DSLR_SCALE), 3))

    i = 0
    for img in TRAIN_IMAGES:

        I = misc.imread(test_directory_orig + str(img) + '.jpg')
        I_depth = misc.imread(test_directory_orig_depth + str(img) + '.png')

        # Downscaling the image by a factor of 2
        I = misc.imresize(I, 0.5, interp='bicubic')

        # Making sure that its width is multiple of 32
        new_width = int(I.shape[1] / 32) * 32
        I = I[:, 0:new_width, :]

        # Stacking the image together with its depth map
        I_temp = np.zeros((I.shape[0], I.shape[1], 4))
        I_temp[:, :, 0:3] = I
        I_temp[:, :, 3] = I_depth
        I = I_temp

        h, w, d = I.shape
        y = np.random.randint(0, w - 512)

        # Extracting random patch of width PATCH_WIDTH
        I = np.float32(I[:, y:y + PATCH_WIDTH, :]) / 255.0
        test_data[i, :] = I

        I = misc.imread(test_directory_blur + str(img) + '.jpg')
        I = np.float32(misc.imresize(I[:, y * 2:y * 2 + 1024, :], DSLR_SCALE / 2, interp='bicubic')) / 255.0
        test_answ[i, :] = I

        i += 1

    return test_data, test_answ


def load_input_image(image_dir, depth_maps_dir, photo):

    I = misc.imread(image_dir + photo)
    I_depth = misc.imread(depth_maps_dir + str(photo.split(".")[0]) + '.png')

    # Downscaling the image by a factor of 2
    I = misc.imresize(I, 0.5, interp='bicubic')

    # Making sure that its width is multiple of 32
    new_width = int(I.shape[1] / 32) * 32
    I = I[:, 0:new_width, :]
    I_depth = I_depth[:, 0:new_width]

    # Stacking the image together with its depth map
    I_temp = np.zeros((I.shape[0], I.shape[1], 4))
    I_temp[:, :, 0:3] = I
    I_temp[:, :, 3] = I_depth

    I = np.float32(I_temp) / 255.0
    I = np.reshape(I, [1, I.shape[0], I.shape[1], 4])

    return I

