from tensorflow.keras.losses import Loss, BinaryCrossentropy, sparse_categorical_crossentropy, MSE, binary_crossentropy
from tensorflow.keras import backend as K

class WholeOutputLoss(Loss):
    '''
    Inspired by https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html
    '''

    def call(self, y_true, y_pred):
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
        true_box_class = y_true[..., 5:]

        pred_box_coords = y_pred[..., 0:2]
        pred_box_size = y_pred[..., 2:4]
        pred_box_x = y_pred[..., 0]
        pred_box_y = y_pred[..., 1]
        pred_box_w = y_pred[..., 2]
        pred_box_h = y_pred[..., 3]
        pred_box_conf = y_pred[..., 4]
        pred_box_class =  y_pred[..., 5:]

        box_conf_loss = 0.5 * K.sum(K.abs(1 - true_box_conf) * K.square(true_box_conf - pred_box_conf)) + \
                        K.sum(true_box_conf * K.square(true_box_conf - pred_box_conf))

        x_loss = K.sum(true_box_conf * K.square(true_box_x - pred_box_x) * K.sqrt(true_box_w))
        y_loss = K.sum(true_box_conf * K.square(true_box_y - pred_box_y) * K.sqrt(true_box_h))
        box_pos_loss = 5 * (x_loss + y_loss)
        box_size_loss = 5 * K.sum(
            true_box_conf * K.sum(K.square(K.sqrt(true_box_sizes) - K.sqrt(pred_box_size)), axis=-1))
        class_loss =  K.sum(true_box_conf *K.sum(K.square(true_box_class - pred_box_class), axis=-1))
        return box_pos_loss + box_size_loss + box_conf_loss + class_loss
    #  return K.sum(true_box_conf * sum_squared_error(y_true, y_pred), axis=-1) \
    #  + K.sum(true_box_conf * sum_squared_sqrt_error(y_true, y_pred), axis=-1) \
    #  + binary_crossentropy(true_box_conf, pred_box_conf)

#
# class IsObjectLoss(Loss):
#     def call(self, y_true, y_pred):
#         y_pred = ops.convert_to_tensor(y_pred)
#         y_true = math_ops.cast(y_true, y_pred.dtype)
#         y_true = K.expand_dims(y_true)
#         loss = K.square(y_true - y_pred)
#         coef = (K.cast(K.equal(y_true, K.zeros_like(y_true)), K.floatx()) + 1) / 2
#         return K.sum(coef * loss, axis=-1)
#
#
# class SumSquaredError(Loss):
#     def call(self, y_true, y_pred, sample_weight=None):
#         if not K.is_keras_tensor(y_pred):
#             y_pred = K.constant(y_pred)
#         y_true = K.cast(y_true, y_pred.dtype)
#         return K.sum(K.square(y_true - y_pred), axis=-1)
#
#
# class SumSquaredSqrtError(Loss):
#     def call(self, y_true, y_pred, sample_weight=None):
#         if not K.is_keras_tensor(y_pred):
#             y_pred = K.constant(y_pred)
#         y_true = K.cast(y_true, y_pred.dtype)
#         return K.sum(K.square(K.sqrt(y_true) - K.sqrt(y_pred)), axis=-1)
