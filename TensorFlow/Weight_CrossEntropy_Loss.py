import tensorflow as tf
from tensorflow.keras import backend as K


def WCELoss(weights, y_true, y_pred):
    pt0 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
    pt1 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
    return (1 - weights) * K.binary_crossentropy(y_true, y_pred) * pt0 + weights * K.binary_crossentropy(y_true,
                                                                                                         y_pred) * pt1
