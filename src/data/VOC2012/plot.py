from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import os
import numpy as np
import tensorflow as tf

def plot_grid(data, df_id, prediction, config, plot_ground_truth=True, highlight_fp=False, plot_fn=False, default_color='g', linewidth=3):
    im = Image.open(os.path.join('data/raw/VOC2012/JPEGImages', data['filename'][df_id]))

    plt.imshow(im)
    # Get the current reference
    ax = plt.gca()

    img_width = data['width'][df_id]
    img_height = data['height'][df_id]

    cell_w = img_width // config.GRID_SIZE[0]
    cell_h = img_height // config.GRID_SIZE[1]


    for i in range(config.GRID_SIZE[0]):

        ax.axvline(i * cell_w, linestyle='--', color='k')  # vertical lines
        for j in range(config.GRID_SIZE[1]):
            pred = prediction[i * config.GRID_SIZE[0] + j]

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
                if (obj > 0.5):
                    color = default_color  # true positive
                    if is_obj == 0 and highlight_fp:
                        color = 'r'  # false positive
                else:
                    color = 'y'  # false negative
                if (color == 'g') or (color == 'r') or (color == 'y' and plot_fn):
                    #print(data['grid_output'][df_id][0][i * GRID_SIZE[0] + j])
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
                    rect_pred = Rectangle((xmin, ymin), width, height, fill=False, linewidth=linewidth,
                                     edgecolor=color)
                    # plt.text(xmin, ymin, str(class_encoder.inverse_transform([np.argmax(cls)])[0]) + '_' + str(
                    #    round(cls[np.argmax(cls)], 2)), bbox=dict(facecolor='red', alpha=0.5))
                    plt.text(x, y + img_height // 16, str(round(obj, 3)), bbox=dict(facecolor=color, alpha=0.5))
                    c = Circle((x, y), radius=5, color=color)
                    ax.add_patch(c)
                    ax.add_patch(rect_pred)

def plot_grid_nms(data, df_id, prediction, config, plot_ground_truth=True, plot_grid=True, linewidth=3, conf_threshold=0.5, iou_threshold=0.5):
    im = Image.open(os.path.join('data/raw/VOC2012/JPEGImages', data['filename'][df_id]))

    plt.imshow(im)
    # Get the current reference
    ax = plt.gca()

    img_width = data['width'][df_id]
    img_height = data['height'][df_id]

    cell_w = img_width // config.GRID_SIZE[0]
    cell_h = img_height // config.GRID_SIZE[1]


    boxes_coords = np.zeros_like(prediction[..., 0:4])

    for i in range(config.GRID_SIZE[0]):
        for j in range(config.GRID_SIZE[1]):
            pred = prediction[i*config.GRID_SIZE[0] + j]
            w =  pred[..., 2] * config.GRID_SIZE[0]
            h = pred[..., 3] * config.GRID_SIZE[1]

            #tensorflow NMS uses (y,x) system i*config.GRID_SIZE[0]][
            boxes_coords[..., 1] = (pred[..., 0] + i) - w/2
            boxes_coords[..., 0] = (pred[..., 1] + j)  - h/2
            boxes_coords[..., 3] = boxes_coords[..., 1] + w
            boxes_coords[..., 2] = boxes_coords[..., 0] + h


    nms_indices = tf.image.non_max_suppression(boxes_coords, prediction[..., 4], max_output_size = config.GRID_SIZE[0]*config.GRID_SIZE[1], score_threshold=conf_threshold, iou_threshold=iou_threshold)
    for i in range(config.GRID_SIZE[0]):

        if plot_grid: ax.axvline(i * cell_w, linestyle='--', color='k')  # vertical lines
        for j in range(config.GRID_SIZE[1]):
            pred = prediction[i * config.GRID_SIZE[0] + j]

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
                                  linewidth=2, edgecolor='b')
                c = Circle((x, y), radius=3, color='b')
                ax.add_patch(c)
                ax.add_patch(rect_true)

            if plot_grid: ax.axhline(j * cell_h, linestyle='--', color='k')  # horizontal lines

            if (i * config.GRID_SIZE[0] + j) in nms_indices:

                width = boxes_size[0]
                height = boxes_size[1]

                x = boxes_coord[0] * cell_w + i * cell_w
                y = boxes_coord[1] * cell_h + j * cell_h

                width *= img_width
                height *= img_height
                xmin = x - width / 2
                ymin = y - height / 2
                # print(xmin, ymin, width, height )
                rect_pred = Rectangle((xmin, ymin), width, height, fill=False, linewidth=linewidth,
                                 edgecolor='r')
                # plt.text(xmin, ymin, str(class_encoder.inverse_transform([np.argmax(cls)])[0]) + '_' + str(
                #    round(cls[np.argmax(cls)], 2)), bbox=dict(facecolor='red', alpha=0.5))
                plt.text(xmin, ymin, str(round(obj, 3)), bbox=dict(facecolor='r', alpha=0.5))
                c = Circle((x, y), radius=3, color='r')
                ax.add_patch(c)
                ax.add_patch(rect_pred)