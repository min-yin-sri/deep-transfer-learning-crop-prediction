import tensorflow as tf
import os
import numpy as np
import pdb
import glob


def batch_reader(img_names, index, read_dir, labels_df, batch_size=64):
    """ Gets the names of the files and ground truth for images and converts them
        to a tf object.
    """
    img_tensor = []
    ground_truth = []
    indexes = img_names.index[index:index+batch_size]
    for index in enumerate(indexes):
        feature_name = img_names[index] + ".npy"
        feature_file_name = os.path.join( read_dir, feature_name )
        img_tensor.append(np.load(feature_file_name))
        ground_truth.append(labels_df[index])

    return img_tensor, np.array(ground_truth)
