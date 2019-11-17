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
