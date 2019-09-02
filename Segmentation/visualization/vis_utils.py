from builtins import range
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

from math import sqrt, ceil
import numpy as np
import tensorflow as tf


def visualize_pairs( x_train_filenames,y_train_filenames,num = 5 ):
        display_num = num

        r_choices = np.random.choice(len(x_train_filenames), display_num)

        plt.figure(figsize=(10, 15))
        for i in range(0, display_num * 2, 2):
            img_num = r_choices[i // 2]
            x_pathname = x_train_filenames[img_num]
            y_pathname = y_train_filenames[img_num]

            plt.subplot(display_num, 2, i + 1)
            plt.imshow(mpimg.imread(x_pathname))
            plt.title("Original Image")

            example_labels = Image.open(y_pathname)
            label_vals = np.unique(example_labels)

            plt.subplot(display_num, 2, i + 2)
            plt.imshow(example_labels)
            plt.title("Masked Image")

        plt.suptitle("Examples of Images and their Masks")
        plt.show()

def visualize_result_triples(helper,weights_path, val_ds):
    # Let's visualize some of the outputs
    data_aug_iter = val_ds.make_one_shot_iterator()
    next_element = data_aug_iter.get_next()

    # Running next element in our graph will produce a batch of images
    plt.figure(figsize=(10, 20))
    for i in range(1):
        batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
        img = batch_of_imgs[0]
        predicted_label = helper.run_inference(weights_path, batch_of_imgs)[0]

        plt.subplot(5, 3, 3 * i + 1)
        plt.imshow(img)
        plt.title("Input image")

        plt.subplot(5, 3, 3 * i + 2)
        plt.imshow(label[0, :, :, 0])
        # plt.imshow(label[0, :, :, :])
        plt.title("Actual Mask")
        plt.subplot(5, 3, 3 * i + 3)
        plt.imshow(predicted_label[:, :, 0])
        # plt.imshow(predicted_label[:, :, :])
        plt.title("Predicted Mask")
    plt.suptitle("Examples of Input Image, Label, and Prediction")
    plt.show()



