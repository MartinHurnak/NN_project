from PIL import  Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from src.helpers import get_bb_min_max
import os
import numpy as np
from src.data.VOC2012.data import class_encoder
from config import GRID_SIZE

def plot(data, df_id, boxes_coord, boxes_size, obj, cls):
    im = Image.open(os.path.join('data/raw/VOC2012/JPEGImages', df['filename'][df_id]))

    plt.imshow(im)
    # Get the current reference
    ax = plt.gca()

    img_width = data['width'][df_id]
    img_height = data['height'][df_id]

    for i in range(len(boxes_coord)):
        width_true = data['output'][df_id][i][2]
        height_true = data['output'][df_id][i][3]
        is_obj = data['output'][df_id][i][4]
        if is_obj == 1:
            xmin_true, ymin_true, _, _ = get_bb_min_max(data['output'][df_id][i][0], data['output'][df_id][i][1],
                                                        width_true, height_true, img_width, img_height)
            # print(xmin_true, ymin_true)
            cls_true = data['class'][df_id][i]
            rect2 = Rectangle((xmin_true, ymin_true), width_true * img_width, height_true * img_height, fill=False,
                              linewidth=3, edgecolor='b')
            plt.text(xmin_true, ymin_true, str(class_encoder.inverse_transform([np.argmax(cls_true)])[0]),
                     bbox=dict(facecolor='blue', alpha=0.5))
            # plt.text(xmin, ymin+height, str(round(o[0],3)), bbox=dict(facecolor='red', alpha=0.5))
            ax.add_patch(rect2)

        if obj[i] > 0.5:
            width = boxes_size[i][0]
            height = boxes_size[i][1]

            xmin, ymin, _, _ = get_bb_min_max(boxes_coord[i][0], boxes_coord[i][1], width, height, img_width,
                                              img_height)
            print(xmin, ymin, obj[i])

            rect = Rectangle((xmin, ymin), width * img_width, height * img_height, fill=False, linewidth=3,
                             edgecolor='r')
            plt.text(xmin, ymin, str(class_encoder.inverse_transform([np.argmax(cls[i])])[0]) + '_' + str(
                round(cls[i][np.argmax(cls[i])], 2)), bbox=dict(facecolor='red', alpha=0.5))
            plt.text(xmin, ymin + height * img_height, str(round(obj[i], 3)), bbox=dict(facecolor='red', alpha=0.5))
            ax.add_patch(rect)

    plt.show()


def plot_grid(data, df_id, prediction):
    im = Image.open(os.path.join('data/raw/VOC2012/JPEGImages', data['filename'][df_id]))
    plt.imshow(im)
    # Get the current reference
    ax = plt.gca()

    img_width = data['width'][df_id]
    img_height = data['height'][df_id]

    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            pred = prediction[i*GRID_SIZE[0]+j]
            boxes_coord = pred[0:2]
            boxes_size =pred[2:4]
            obj = pred[5]
            cls = pred[:20]

            print(boxes_coord, boxes_size, obj, cls)

            # width_true = data['output'][df_id][i][2]
            # height_true = data['output'][df_id][i][3]
            # is_obj = data['output'][df_id][i][4]
            # if is_obj == 1:
            #     xmin_true, ymin_true, _, _ = get_bb_min_max(data['output'][df_id][i][0], data['output'][df_id][i][1],
            #                                                 width_true, height_true, img_width, img_height)
            #     # print(xmin_true, ymin_true)
            #     cls_true = data['class'][df_id][i]
            #     rect2 = Rectangle((xmin_true, ymin_true), width_true * img_width, height_true * img_height, fill=False,
            #                       linewidth=3, edgecolor='b')
            #     plt.text(xmin_true, ymin_true, str(class_encoder.inverse_transform([np.argmax(cls_true)])[0]),
            #              bbox=dict(facecolor='blue', alpha=0.5))
            #     # plt.text(xmin, ymin+height, str(round(o[0],3)), bbox=dict(facecolor='red', alpha=0.5))
            #     ax.add_patch(rect2)

            if obj >= 0.0:
                width = boxes_size[0]
                height = boxes_size[1]
                cell_w = img_width // GRID_SIZE[0]
                cell_h = img_height // GRID_SIZE[1]

                x = boxes_coord[0] * cell_w + i * cell_w
                y = boxes_coord[1] * cell_h + i * cell_h
                xmin = x- width/2
                ymin = y-width/2
               # xmin, ymin, _, _ = get_bb_min_max(x, y, width, height, img_width,img_height)
                print(xmin, ymin, obj)

                rect = Rectangle((xmin, ymin), width * img_width, height * img_height, fill=False, linewidth=3,
                                 edgecolor='r')
                plt.text(xmin, ymin, str(class_encoder.inverse_transform([np.argmax(cls)])[0]) + '_' + str(
                    round(cls[np.argmax(cls)], 2)), bbox=dict(facecolor='red', alpha=0.5))
                plt.text(xmin, ymin + height * img_height, str(round(obj, 3)), bbox=dict(facecolor='red', alpha=0.5))
                ax.add_patch(rect)

    plt.show()



