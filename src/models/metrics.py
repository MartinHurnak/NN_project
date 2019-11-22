import tensorflow.keras.backend as K
from config import GRID_SIZE
from src.helpers import intersection_over_union

def true_positives(y_true, y_pred):
    # print(y_pred[...,-1])
    # return K.zeros_like(y_pred)

    if not K.is_keras_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    true_box_coords = y_true[..., 0:2]
    true_box_sizes = y_true[..., 2:4]
    true_box_x = y_true[..., 0]
    true_box_y = y_true[..., 1]
    true_box_w = y_true[..., 2]
    true_box_h = y_true[..., 3]
    true_box_conf = y_true[..., 4]
    # true_box_class = y_true[..., 5:]

    pred_box_coords = y_pred[..., 0:2]
    pred_box_size = y_pred[..., 2:4]
    pred_box_x = y_pred[..., 0]
    pred_box_y = y_pred[..., 1]
    pred_box_w = y_pred[..., 2]
    pred_box_h = y_pred[..., 3]
    pred_box_conf = y_pred[..., 4]
    # pred_box_class =  y_pred[..., 5:]
    for i in range(16):
        iou = intersection_over_union(GRID_SIZE, pred_box_x, pred_box_y, pred_box_w, pred_box_h,
                                      K.expand_dims(true_box_x[..., i]), K.expand_dims(true_box_y[..., i]),
                                      K.expand_dims(true_box_w[..., i]), K.expand_dims(true_box_h[..., i]))
        tp = K.cast(K.greater_equal(iou, 0.5 * K.ones_like(iou)), 'float32') * (
            K.cast(K.greater_equal(pred_box_conf, 0.5 * K.ones_like(pred_box_conf)), 'float32'))
        print(tp)
    return tp
