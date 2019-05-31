""" This script trains a CNN model

    Example :
    python train.py
    --train_dir ../Aerial_Vehicle_Detection/train_dirsig
    --test_dir ../Aerial_Vehicle_Detection/train_dirsig
    --training_data_dir ./debug.csv
    --labels_dir ./categories.csv
    --batch_size 32
    --learning_rate 1e-4
    --test_frequency 10
    --number_iterations 20
    --ckpt_dir ./ckpt_dir/model
"""

import tensorflow as tf
import numpy as np
import argparse
import prepare_dataset
#import pandas as pd
import pdb
import logging
from tqdm import tqdm
import os
import glob
import csv


from model import PovertyMapper

PATH = "/root/bucket3/textual_global_feature_vectors"
TRAINING_PATH = "/root/bucket3/textual_global_feature_vectors/training_sets"
ETHIOPIA_GROUD_TRUTH_FILENAME = "Ethiopia_Grouth_Truth.csv"

logging.basicConfig(filename='train.log',level=logging.DEBUG)

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--batch_size', type=int, default=32,
                         help='Batch size in the training stage')
    aparser.add_argument('--number_epochs', type=int, default=100,
                         help='Number of epochs to train the network')
    aparser.add_argument('--learning_rate', type=float, default=1e-4,
                         help='Learning rate')
    aparser.add_argument('--test_frequency', type=int, default=10,
                         help='After every provided number of iterations the model will be test')
    aparser.add_argument('--train_dir', type=str, default=TRAINING_PATH,
                         help='Provide the training directory to the text file with file names and labels in it')
    aparser.add_argument('--test_dir', type=str, default=TRAINING_PATH,
                     help='Provide the test directory to the text file with file names and labels in it')
    aparser.add_argument('--ckpt_dir', type=str,
                     help='Provide the checkpoint directory where the network parameters will be stored')
    aparser.add_argument('--training_data_dir', type=str, default=TRAINING_PATH,
                     help='Provide the directory where the image names and labels are stored')
    aparser.add_argument('--labels_dir', type=str,
                     help='Provide the directory where the mapping function is stored for the labels')
    return aparser

def main():

    # Parse the command line args
    args = get_parser().parse_args()

    # Read the filenames
    ground_truth_input_file = os.path.join( PATH, ETHIOPIA_GROUD_TRUTH_FILENAME )
    logging.info("Ground truth file is at %s" % ground_truth_input_file)
    with open(ground_truth_input_file, 'rb') as gf:
        greader = csv.reader(gf)
        ground_truth_list = list(greader)
    logging.info("ground truth csv file has %d entries" % len(ground_truth_list))
    logging.info("The first line of ground truth csv file: %s %s" % (ground_truth_list[0][7], ground_truth_list[0][12]) )
    logging.info("The first line of ground truth csv file: %s %s" % (ground_truth_list[1][7], ground_truth_list[1][12]) )

    train_filenames = []
    labels_df = []
    ground_truth_index = 0
    for ground_truth_entry in ground_truth_list:
        if ground_truth_index == 0:
            ground_truth_index = ground_truth_index + 1
            continue
        train_filenames.append(ground_truth_entry[7])
        labels_df.append(ground_truth_entry[12])

    #ratio_validation = int(len(train_filenames) / 5)
    #val_filenames = train_filenames[:ratio_validation]
    #train_filenames = train_filenames[ratio_validation:]
    val_filenames = train_filenames

    # Build the Graph
    net = PovertyMapper()

    x_train = tf.placeholder(tf.float32, [None, 3000], name="input")
    y_train = tf.placeholder(tf.float32, [None, None], name="target")
    is_training = tf.placeholder(tf.bool, name="batch_norm_bool")

    # Construct the network
    num_labels = len(labels_df)
    y_pred = net.graph(x_train, y_train, is_training, num_labels)

    # Define the loss function and optimizer
    cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(y_train, y_pred))
    optimizer = tf.train.AdamOptimizer(args.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         train_op = optimizer.minimize(cross_entropy, global_step=tf.train.get_global_step())

    # Save all the variables
    saver = tf.train.Saver()

    # Execute the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_counter = 0
        for epoch_number in tqdm(range(args.number_epochs)):
            for epoch_id, iteration_number in enumerate(range(0, len(train_filenames), args.batch_size)):
                # Train the model on training dataset
                try:
                    # Update the batch reader to read 1D numpy arrays
                    ### ------ This Batch Reader Needs to Return Bx3000 D tensor and corresponding labels Bx1 ------ #####
                    train_imgs, train_labels = prepare_dataset.batch_reader(train_filenames, iteration_number, args.train_dir, labels_df, args.batch_size)
                    _, loss_value = sess.run([train_op, cross_entropy], feed_dict={is_training : True, x_train: np.reshape(train_imgs, [32, 3000]), y_train: np.reshape(train_labels, [32,1])})
                    train_counter += 1
                except Exception as error:
                    continue
                if epoch_id % 50 == 0:
                    print("Training Loss at iteration {} {} : {}".format(epoch_number, epoch_id, loss_value))

            # Test the model on validation dataset at the end of every epoch
            #pdb.set_trace()
            overall_loss_value, counter = 0., 0.
            for iteration_number in range(0, len(val_filenames), args.batch_size):
                try:
                    val_imgs, val_labels = prepare_dataset.batch_reader(val_filenames, iteration_number, args.test_dir, labels_df, args.batch_size)
                    loss_value = sess.run(cross_entropy, feed_dict={is_training : False, x_train: np.reshape(val_imgs, [32, 3000]), y_train: np.reshape(val_labels, [32,1])})
                    overall_loss_value += loss_value
                    counter += 1.
                except Exception as error:
                    print(error)
                    continue
            print("Validation Loss at iteration {} : {}".format(epoch_number, overall_loss_value / counter))

if __name__ == '__main__':
    main()
