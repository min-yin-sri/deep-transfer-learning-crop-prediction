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
    indexes = img_names[index:index+batch_size]
    for counter, index in enumerate(indexes):
        feature_name = index + ".npy"
        feature_file_name = os.path.join( read_dir, feature_name )
        img_tensor.append(np.load(feature_file_name))
        ground_truth.append(labels_df[counter])

    return np.array(img_tensor), np.array(ground_truth)
