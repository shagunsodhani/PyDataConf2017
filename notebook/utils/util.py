import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, merge, BatchNormalization, Activation, Input, Merge
from keras import backend as K

def get_glove_embeddings(file_path):
    '''Method to read the GloVe embeddings'''
    embeddings_index = {}
    with open(file_path) as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index


def euclidean_distance(vects):
    return K.sqrt(K.sum(K.square(vects[0] - vects[1]), axis=1, keepdims=True))

def eucl_dist_output_shape(shapes):
    return (shapes[0][0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 2
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    '''
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))