def get_center_size(xmin, ymin, xmax, ymax, img_width, img_height):
    if xmin > xmax:
        raise AttributeError('{} > {}'.format(xmin, xmax))
    if ymin > ymax:
        raise AttributeError('{} > {}'.format(ymin, ymax))

    xmin = 2 * xmin / img_width - 1
    ymin = 2 * ymin / img_height - 1
    xmax = 2 * xmax / img_width - 1
    ymax = 2 * ymax / img_height - 1

    w = abs(xmax - xmin)
    h = abs(ymax - ymin)

    x = xmin + w / 2
    y = ymin + h / 2

    return x, y, w / 2, h / 2


def get_bb_min_max(x, y, w, h, img_width, img_height):
    w *= 2
    h *= 2

    xmin = x - w / 2
    ymin = y - h / 2
    xmax = x + w / 2
    ymax = y + h / 2

    xmin = round((xmin + 1) * img_width / 2)
    ymin = round((ymin + 1) * img_height / 2)
    xmax = round((xmax + 1) * img_width / 2)
    ymax = round((ymax + 1) * img_height / 2)

    return xmin, ymin, xmax, ymax
