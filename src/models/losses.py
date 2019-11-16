from tensorflow.keras.losses import Loss
from tensorflow.python import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K


class SumSquaredError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.sum(K.square(y_true - y_pred), axis=-1)


class SumSquaredSqrtError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.sum(K.square(K.sqrt(y_true) - K.sqrt(y_pred)), axis=-1)


class MeanSquaredSqrtError(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(K.square(K.sqrt(y_true) - K.sqrt(y_pred)), axis=-1)
