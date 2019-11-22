from tensorflow.keras import backend as K


def get_center_size(xmin, ymin, xmax, ymax, img_width, img_height):
    if xmin > xmax:
        raise AttributeError('{} > {}'.format(xmin, xmax))
    if ymin > ymax:
        raise AttributeError('{} > {}'.format(ymin, ymax))

    xmin = xmin / img_width
    ymin = ymin / img_height
    xmax = xmax / img_width
    ymax = ymax / img_height

    w = abs(xmax - xmin)
    h = abs(ymax - ymin)

    x = xmin + w / 2
    y = ymin + h / 2

    return x, y, w, h


def get_bb_min_max(x, y, w, h, img_width, img_height):

    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2

    xmin = round(xmin * img_width )
    ymin = round(ymin * img_height)
    xmax = round(xmax * img_width)
    ymax = round(ymax * img_height)

    return xmin, ymin, xmax, ymax

# inspired by: https://blog.emmanuelcaradec.com/humble-yolo-implementation-in-keras/
def intersection_over_union(grid_size, pred_box_x, pred_box_y, pred_box_w, pred_box_h, true_box_x, true_box_y,
                            true_box_w, true_box_h):
    intersect_w = K.maximum(K.zeros_like(pred_box_w), (pred_box_w + true_box_w) * grid_size[0] / 2 - K.abs(
        pred_box_x - true_box_x))
    intersect_h = K.maximum(K.zeros_like(pred_box_h), (pred_box_h + true_box_h) * grid_size[1] / 2 - K.abs(
        pred_box_y - true_box_y))

    intersect_area = intersect_w * intersect_h

    true_area = true_box_w * grid_size[0] * true_box_h * grid_size[1]
    pred_area = pred_box_w * grid_size[0] * pred_box_h * grid_size[1]
    union_area = pred_area + true_area - intersect_area

    return intersect_area / union_area
