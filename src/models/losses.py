from tensorflow.keras.losses import Loss, BinaryCrossentropy, sparse_categorical_crossentropy
from tensorflow.python import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras import backend as K

# class TotalLoss(Loss):
#     def call(self, y_true, y_pred):
#
#         y_is_obj = ops.convert_to_tensor(y_true[...,0:1][0])
#
#         bb_loss = K.sum(K.square(y_true[...,2:6] - y_pred[...,2:6]), axis=-1)
#
#         class_loss = sparse_categorical_crossentropy(y_true[...,-1], y_pred[...,6:])
#         loss = bb_loss + class_loss
#         loss = K.switch(K.equal(y_is_obj,K.zeros_like(y_is_obj)), loss, K.zeros_like(loss))
#         return loss

def class_loss(is_object):
    def loss(y_true, y_pred):
        is_obj = ops.convert_to_tensor(is_object)
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        ls = K.square(y_true - y_pred)
        coef = K.cast(K.equal(is_object, K.zeros_like(is_obj)), K.floatx())
        return K.sum(coef * ls, axis=-1)
    return loss

class IsObjectLoss(Loss):
    def call(self, y_true, y_pred):
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        y_true = K.expand_dims(y_true)
        loss = K.square(y_true - y_pred)
        coef = (K.cast(K.equal(y_true, K.zeros_like(y_true)), K.floatx()) + 1) /2
        return K.sum(coef*loss, axis=-1)


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
