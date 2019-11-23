from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
from src.data.VOC2012.data import class_encoder
from config import GRID_SIZE


def plot_grid(data, df_id, prediction, plot_ground_truth=True, plot_fp=False, plot_fn=False):
    im = Image.open(os.path.join('data/raw/VOC2012/JPEGImages', data['filename'][df_id]))
    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    # Get the current reference
    ax = plt.gca()

    img_width = data['width'][df_id]
    img_height = data['height'][df_id]

    cell_w = img_width // GRID_SIZE[0]
    cell_h = img_height // GRID_SIZE[1]

    for i in range(GRID_SIZE[0]):

        ax.axvline(i * cell_w, linestyle='--', color='k')  # vertical lines
        for j in range(GRID_SIZE[1]):
            pred = prediction[i * GRID_SIZE[0] + j]

            boxes_coord = pred[0:2]
            boxes_size = pred[2:4]
            obj = pred[4]
            # cls = pred[5:]
            true = data['grid_' + str(i) + '_' + str(j)][df_id][0][0]
            is_obj = true[4]

            if (is_obj == 1) and plot_ground_truth:
                width = true[2]
                height = true[3]

                x = true[0] * cell_w + i * cell_w
                y = true[1] * cell_h + j * cell_h

                width *= img_width
                height *= img_height
                xmin = x - width / 2
                ymin = y - height / 2

                rect_true = Rectangle((xmin, ymin), width, height, fill=False,
                                  linewidth=3, edgecolor='b')
                c = Circle((x, y), radius=5, color='b')
                ax.add_patch(c)
                ax.add_patch(rect_true)

            ax.axhline(j * cell_h, linestyle='--', color='k')  # horizontal lines
            if (obj > 0.5) or (is_obj == 1):
                if (obj > 0.5) and (is_obj == 1):
                    color = 'g'  # true positive
                elif is_obj == 0:
                    color = 'r'  # false positive
                else:
                    color = 'y'  # false negative
                if (color == 'g') or (color == 'r' and plot_fp) or (color == 'y' and plot_fn):
                    #print(i, j, boxes_coord, boxes_size, obj, is_obj)
                    width = boxes_size[0]
                    height = boxes_size[1]

                    x = boxes_coord[0] * cell_w + i * cell_w
                    y = boxes_coord[1] * cell_h + j * cell_h

                    width *= img_width
                    height *= img_height
                    xmin = x - width / 2
                    ymin = y - height / 2

                    # print(xmin, ymin, width, height )
                    rect_pred = Rectangle((xmin, ymin), width, height, fill=False, linewidth=5,
                                     edgecolor=color)
                    # plt.text(xmin, ymin, str(class_encoder.inverse_transform([np.argmax(cls)])[0]) + '_' + str(
                    #    round(cls[np.argmax(cls)], 2)), bbox=dict(facecolor='red', alpha=0.5))
                    plt.text(x, y + img_height // 16, str(round(obj, 3)), bbox=dict(facecolor=color, alpha=0.5))
                    c = Circle((x, y), radius=5, color=color)
                    ax.add_patch(c)
                    ax.add_patch(rect_pred)

    plt.show()
