from tensorflow.keras.metrics import Metric
import numpy as np
import tensorflow.keras.backend as K

def test_met(y_true, y_pred):
    #print(y_pred[...,-1])
    #return K.zeros_like(y_pred)

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
        argm = iou([4,4], pred_box_x, pred_box_y, pred_box_w, pred_box_h, K.expand_dims(true_box_x[...,i]), K.expand_dims(true_box_y[...,i]), K.expand_dims(true_box_w[...,i]), K.expand_dims(true_box_h[...,i]))
        #print(argm)
        #print(K.greater_equal(argm, 0.5*K.ones_like(argm)))
        print(K.cast(K.greater_equal(argm, 0.5*K.ones_like(argm)), 'float32') * (K.cast(K.greater_equal(pred_box_conf, 0.5 * K.ones_like(pred_box_conf)), 'float32')))
        print(
        #print(true_box_conf[...,i])
        #print(K.equal(true_box_conf[...,i]), K.ones_like())
    return K.argmax(iou([4,4], pred_box_x, pred_box_y, pred_box_w, pred_box_h, true_box_x, true_box_y, true_box_w, true_box_h))

def iou(grid_size, pred_box_x, pred_box_y, pred_box_w, pred_box_h, true_box_x, true_box_y, true_box_w, true_box_h):
    intersect_w = K.maximum(K.zeros_like(pred_box_w), (pred_box_w + true_box_w) * grid_size[0] / 2 - K.abs(
        pred_box_x - true_box_x))
    intersect_h = K.maximum(K.zeros_like(pred_box_h), (pred_box_h + true_box_h) * grid_size[1] / 2 - K.abs(
        pred_box_y - true_box_y))

    intersect_area = intersect_w * intersect_h

    true_area = true_box_w * grid_size[0] * true_box_h * grid_size[1]
    pred_area = pred_box_w * grid_size[0] * pred_box_h * grid_size[1]
    union_area = pred_area + true_area - intersect_area

    return intersect_area / union_area


class Met(Metric):
    def __init__(self):
        super(Met, self).__init__('metric', np.float32)
        pass

    def update_state(self, y_true, y_pred):
        print(y_pred[...,-1])

    def add_weight(self):
        pass

    def reset_states(self):
        pass

    def result(self):
        pass
