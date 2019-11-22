from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K
from config import GRID_SIZE
from src.helpers import intersection_over_union

class SumSquaredLoss(Loss):
    '''
    Inspired by https://fairyonice.github.io/Part_4_Object_Detection_with_Yolo_using_VOC_2012_data_loss.html
    '''

    def __init__(self, grid_size=GRID_SIZE, negative_box_coef=0.25, position_coef=5, size_coef=5):
        super(SumSquaredLoss, self).__init__()

        self.grid_size = grid_size
        self.negative_box_coef = negative_box_coef
        self.position_coef = position_coef
        self.size_coef = size_coef

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
        # true_box_class = y_true[..., 5:]

        pred_box_coords = y_pred[..., 0:2]
        pred_box_size = y_pred[..., 2:4]
        pred_box_x = y_pred[..., 0]
        pred_box_y = y_pred[..., 1]
        pred_box_w = y_pred[..., 2]
        pred_box_h = y_pred[..., 3]
        pred_box_conf = y_pred[..., 4]
        # pred_box_class =  y_pred[..., 5:]


        iou = intersection_over_union(self.grid_size, pred_box_x, pred_box_y, pred_box_w, pred_box_h, true_box_x, true_box_y, true_box_w, true_box_h)

        conf_loss = K.sum(K.square(true_box_conf * iou - pred_box_conf) * true_box_conf, axis=-1)

        conf_loss = conf_loss + self.negative_box_coef * K.sum(
            K.square(true_box_conf * iou - pred_box_conf) * K.abs(1 - true_box_conf), axis=-1)

        box_pos_loss = self.position_coef * K.sum(
            true_box_conf * K.sum(K.square(true_box_coords - pred_box_coords), axis=-1), axis=-1)

        box_size_loss = self.size_coef * K.sum(
            true_box_conf * K.sum(K.square(K.sqrt(true_box_sizes+K.epsilon()) - K.sqrt(pred_box_size+K.epsilon())), axis=-1), axis=-1)
        # class_loss =  K.sum(true_box_conf *K.sum(K.square(true_box_class - pred_box_class), axis=-1))
        loss = box_pos_loss + box_size_loss + conf_loss  # + class_loss
        return loss
