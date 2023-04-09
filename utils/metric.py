from keras import backend as K
from scipy.stats import spearmanr
import tensorflow as tf
def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)


def get_spearman_rankcor(y_true, y_pred):
    return (tf.py_function(
        spearmanr, [tf.cast(y_pred, tf.float32),
                    tf.cast(y_true, tf.float32)],
        Tout=tf.float32))