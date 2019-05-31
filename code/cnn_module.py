import tensorflow as tf
import numpy as np
import pdb

class CNN():
    """ This class contains the components of any CNN Architecture """
    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def weight_variable(self, shape, filter_name):
        """ Define the Weights and Initialize Them and Attach to the Summary """
        weights = tf.get_variable(filter_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variable_summaries(weights)
        return weights

    def bias_variable(self, shape, bias_name):
        """ Define the Biases and Initialize Them and Attach to the Summary """
        bias = tf.get_variable(bias_name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.variable_summaries(bias)
        return bias

    def _batch_norm(self, input, filter_id, is_training):
        """ Apply Batch Normalization After Convolution and Before Activation """
        input_norm = tf.layers.batch_normalization(input, momentum=0.95, center=True, scale=True, training=is_training, name='bn'+str(filter_id))
        return input_norm

    def _fcl(self, input_data, shape, bias_shape, filter_id, classification_layer=False):
        """ Run a Fully Connected Layer and ReLU if necessary """
        weights = self.weight_variable(shape, 'weights'+  filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)
        out_fc_layer = tf.reshape(input_data, [-1, shape[0]])
        predictions = tf.add(tf.matmul(out_fc_layer, weights), bias, name="predictions_op")
        predictions = tf.nn.relu(predictions)
        return predictions

    def _regression_layer(self, input_data, shape, bias_shape, filter_id, classification_layer=False):
        """ Run a Fully Connected Layer and ReLU if necessary """
        weights = self.weight_variable(shape, 'weights'+  filter_id)
        bias = self.bias_variable(bias_shape, 'bias' + filter_id)
        out_fc_layer = tf.reshape(input_data, [-1, shape[0]])
        predictions = tf.add(tf.matmul(out_fc_layer, weights), bias, name="predictions_op")
        return predictions