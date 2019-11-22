import tensorflow.keras.backend as K
from config import GRID_SIZE
from src.helpers import intersection_over_union


def precision(y_true, y_pred):
    tp = true_positives(y_true, y_pred)
    p = positives(y_true, y_pred)
    return tp / (p+K.epsilon())


def positives(y_true, y_pred):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)

    pred_box_conf = y_pred[..., 4]

    p_box = (K.cast(K.greater(pred_box_conf, 0.5 * K.ones_like(pred_box_conf)), 'float32'))
    return K.sum(p_box)


def true_positives(y_true, y_pred):
    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    true_box_x = y_true[..., 0]
    true_box_y = y_true[..., 1]
    true_box_w = y_true[..., 2]
    true_box_h = y_true[..., 3]

    pred_box_x = y_pred[..., 0]
    pred_box_y = y_pred[..., 1]
    pred_box_w = y_pred[..., 2]
    pred_box_h = y_pred[..., 3]
    pred_box_conf = y_pred[..., 4]
    tp = K.zeros(1)
    for i in range(16):
        iou = intersection_over_union(GRID_SIZE, pred_box_x, pred_box_y, pred_box_w, pred_box_h,
                                      K.expand_dims(true_box_x[..., i]), K.expand_dims(true_box_y[..., i]),
                                      K.expand_dims(true_box_w[..., i]), K.expand_dims(true_box_h[..., i]))
        tp_cell = K.cast(K.greater(iou, 0.5 * K.ones_like(iou)), 'float32') * (
            K.cast(K.greater(pred_box_conf, 0.5 * K.ones_like(pred_box_conf)), 'float32'))
        tp_cell = K.max(tp_cell, axis=-1)
        tp = tp + tp_cell
    return tp
