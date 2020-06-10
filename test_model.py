# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from scipy import misc
import numpy as np
import tensorflow as tf
import sys
import os

tf.compat.v1.disable_v2_behavior()

from load_dataset import load_input_image
from model import PyNET
import utils

LEVEL, restore_iter, dataset_dir, use_gpu, orig_model = utils.process_test_model_args(sys.argv)
DSLR_SCALE = float(1) / (2 ** (LEVEL - 2))

# Disable gpu if specified
config = tf.compat.v1.ConfigProto(device_count={'GPU': 0}) if use_gpu == "false" else None


with tf.compat.v1.Session(config=config) as sess:

    # Placeholders for test data
    x_ = tf.compat.v1.placeholder(tf.float32, [1, None, None, 4])
    y_ = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3])

    # generate bokeh image

    output_l1, output_l2, output_l3, output_l4, output_l5, output_l6, output_l7 = \
        PyNET(x_, instance_norm=True, instance_norm_level_1=False)

    if LEVEL == 7:
        bokeh_img = output_l7
    if LEVEL == 6:
        bokeh_img = output_l6
    if LEVEL == 5:
        bokeh_img = output_l5
    if LEVEL == 4:
        bokeh_img = output_l4
    if LEVEL == 3:
        bokeh_img = output_l3
    if LEVEL == 2:
        bokeh_img = output_l2
    if LEVEL == 1:
        bokeh_img = output_l1

    bokeh_img = tf.clip_by_value(bokeh_img, 0.0, 1.0)

    # Losses
    loss_psnr = tf.reduce_mean(tf.image.psnr(bokeh_img, y_, 1.0))
    loss_ssim = tf.reduce_mean(tf.image.ssim(bokeh_img, y_, 1.0))
    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(bokeh_img, y_, 1.0))

    # Loading pre-trained model

    saver = tf.compat.v1.train.Saver()

    if orig_model == "true":
        saver.restore(sess, "models/original/pynet_bokeh_level_0")
    else:
        saver.restore(sess, "models/pynet_level_" + str(LEVEL) + "_iteration_" + str(restore_iter) + ".ckpt")

    # -------------------------------------------------
    # Part 1:  Processing sample full-resolution images

    print("Generating sample visual results")

    sample_images_dir = "visual_samples/images/"
    sample_depth_maps_dir = "visual_samples/depth_maps/"

    sample_images = [f for f in os.listdir(sample_images_dir) if os.path.isfile(sample_images_dir + f)]

    for photo in sample_images:

        # Load image

        I = load_input_image(sample_images_dir, sample_depth_maps_dir, photo)

        # Run inference

        bokeh_tensor = sess.run(bokeh_img, feed_dict={x_: I})
        bokeh_image = np.reshape(bokeh_tensor, [int(I.shape[1] * DSLR_SCALE), int(I.shape[2] * DSLR_SCALE), 3])

        # Save the results as .png images
        photo_name = photo.rsplit(".", 1)[0]
        misc.imsave("results/full-resolution/" + photo_name + "_level_" + str(LEVEL) +
                        "_iteration_" + str(restore_iter) + ".png", bokeh_image)

    # ------------------------------------------------------------------------
    # Part 1:  Compute PSNR / SSIM scores on the test part of the EBB! dataset

    print("Performing quantitative evaluation")

    test_directory_orig = dataset_dir + 'test/original/'
    test_directory_orig_depth = dataset_dir + 'test/original_depth/'
    test_directory_blur = dataset_dir + 'test/bokeh/'

    test_images = [f for f in os.listdir(test_directory_orig) if os.path.isfile(os.path.join(test_directory_orig, f))]

    loss_psnr_ = 0.0
    loss_ssim_ = 0.0
    loss_msssim_ = 0.0

    test_size = len(test_images)
    iter_ = 0

    for photo in test_images:

        # Load image

        I = load_input_image(test_directory_orig, test_directory_orig_depth, photo)

        Y = misc.imread(test_directory_blur + photo) / 255.0
        Y = np.float32(misc.imresize(Y, DSLR_SCALE / 2, interp='bicubic')) / 255.0
        Y = np.reshape(Y, [1, Y.shape[0], Y.shape[1], 3])

        loss_psnr_temp, loss_ssim_temp, loss_msssim_temp = sess.run([loss_psnr, loss_ssim, loss_ms_ssim],
                                                                    feed_dict={x_: I, y_: Y})

        print(photo, iter_, loss_psnr_temp, loss_ssim_temp, loss_msssim_temp)

        loss_psnr_ += loss_psnr_temp / test_size
        loss_ssim_ += loss_ssim_temp / test_size
        loss_msssim_ += loss_msssim_temp / test_size

        iter_ += 1

    output_logs = "PSNR: %.4g, SSIM: %.4g, MS-SSIM: %.4g\n" % (loss_psnr_, loss_ssim_, loss_msssim_)
    print(output_logs)
