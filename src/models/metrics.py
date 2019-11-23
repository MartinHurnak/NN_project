import tensorflow.keras.backend as K
from config import GRID_SIZE
from src.helpers import intersection_over_union
import numpy as np

def recall(y_true, y_pred, threshold=0.5):
    tp = true_positives(y_true, y_pred, threshold)
    fn = false_negative(y_true, y_pred, threshold)
    return tp / (tp + fn)


def precision(y_true, y_pred, threshold=0.5):
    tp = true_positives(y_true, y_pred, threshold)
    p = positives(y_true, y_pred, threshold)
    zero_positives = K.cast(K.equal(p, K.zeros_like(p)),K.floatx())
    return K.maximum(tp / (p+K.epsilon()), zero_positives)


def positives(y_true, y_pred, threshold=0.5):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)

    pred_box_conf = y_pred[..., 4]

    p_box = (K.cast(K.greater(pred_box_conf, threshold * K.ones_like(pred_box_conf)), K.floatx()))

    return K.sum(p_box, axis=-1)


def true_positives(y_true, y_pred, threshold=0.5):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    true_box_x = y_true[..., 0]
    true_box_y = y_true[..., 1]
    true_box_w = y_true[..., 2]
    true_box_h = y_true[..., 3]
    true_box_conf = y_true[...,4]

    pred_box_x = y_pred[..., 0]
    pred_box_y = y_pred[..., 1]
    pred_box_w = y_pred[..., 2]
    pred_box_h = y_pred[..., 3]
    pred_box_conf = y_pred[..., 4]

    iou = intersection_over_union(GRID_SIZE, pred_box_x, pred_box_y, pred_box_w, pred_box_h,
                                  true_box_x, true_box_y,
                                  true_box_w,true_box_h)
    tp = K.cast(K.greater(iou, 0.5 * K.ones_like(iou)), K.floatx()) * (
        K.cast(K.greater(pred_box_conf, threshold * K.ones_like(pred_box_conf)), K.floatx()))
    return K.sum(tp, axis=-1)


def false_negative(y_true, y_pred, threshold=0.5):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    true_box_conf = y_true[...,4]

    fn = K.sum(true_box_conf, axis=-1) - true_positives(y_true, y_pred, threshold)
    return fn
