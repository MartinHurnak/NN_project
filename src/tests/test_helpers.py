from src.helpers import get_center_size, get_bb_min_max


def test_get_center_sizes():
    x, y, w, h = get_center_size(0, 0, 500, 500, 500, 500)
    assert x == 0
    assert y == 0
    assert w == 1
    assert h == 1

    x, y, w, h = get_center_size(0, 0, 250, 250, 500, 500)
    assert x == -0.5
    assert y == -0.5
    assert w == 0.5
    assert h == 0.5

    x, y, w, h = get_center_size(125, 125, 375, 375, 500, 500)
    assert x == 0
    assert y == 0
    assert w == 0.5
    assert h == 0.5

    for i in range(10):
        for j in range(10):
            for k in range(i,10):
                for l in range(j,10):
                    x, y, w, h = get_center_size(i, j, k, l, 9, 9)
                    assert (x >= -1) and (x <= 1)
                    assert (y >= -1) and (y <= 1)
                    assert (w >= 0) and (w <= 1)
                    assert (h >= 0) and (h <= 1)



def test_get_bb_min_max():
    xmin, ymin, xmax, ymax = get_bb_min_max(-1, -1, 1, 1, 500, 500)
    assert xmin == 0
    assert xmax == 500
    assert ymin == 0
    assert ymax == 500

    xmin, ymin, xmax, ymax = get_bb_min_max(-0.5, -0.5, 0.5, 0.5, 500, 500)
    assert xmin == 125
    assert xmax == 125
    assert ymin == 375
    assert ymax == 375

    xmin = 154
    xmax = 246
    ymin = 124
    ymax = 456

    x, y, w, h = get_center_size(xmin, xmax, ymin, ymax, 500, 500)
    xmin2, ymin2, xmax2, ymax2 = get_bb_min_max(x, y, w, h, 500, 500)

    assert xmin == xmin2
    assert xmax == xmax2
    assert ymin == ymin2
    assert ymax == ymax2
