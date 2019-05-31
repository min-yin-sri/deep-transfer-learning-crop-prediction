import tensorflow as tf
import numpy as np
from cnn_module import CNN

import pdb
class PovertyMapper(CNN):

    def graph(self, x_train, y_train, is_training, num_classes):
        out_1 = CNN._fcl(self, x_train, [3000, 1000], [1000], 'fc_1', classification_layer=False)
        out_2 = CNN._fcl(self, out_1, [1000, 500], [500], 'fc_2', classification_layer=False)
        out_3 = CNN._fcl(self, out_2, [500, 250], [250], 'fc_3', classification_layer=False)
        y_pred = CNN._regression_layer(self, out_3, [250, 1], [1], 'fc_4', classification_layer=True)

        return y_pred
