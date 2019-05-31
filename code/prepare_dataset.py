import tensorflow as tf
import os
import numpy as np
import pdb
import glob

from PIL import Image

def batch_reader(img_names, index, read_dir, labels_df, img_shape=[224,224,3], batch_size=64):
    """ Gets the names of the files and ground truth for images and converts them
        to a tf object.
    """
    img_tensor = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]), dtype=np.int32)
    ground_truth = []
    indexes = img_names.index[index:index+batch_size]
    for counter, index in enumerate(indexes):
        img = Image.open("{}{}{}.jpeg".format(read_dir, 'dg_wiki_africa_1000x1000_', img_names['id'][index])).convert("RGB")
        img_tensor[counter, :, :, :] = np.array(img.resize([img_shape[0], img_shape[1]])) - 110.
        one_hot = np.zeros(len(labels_df))
        one_hot[label_merger(labels_df, img_names['category'][index])] = 1
        ground_truth.append(one_hot)

    return img_tensor, np.array(ground_truth)
