from config import SEQUENCE_LENGTH, GRID_SIZE
import numpy as np
from src.data.VOC2012.data import classes, class_encoder
from src.data.VOC2012.data_loader import load_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from src.helpers import get_center_size

from config import GRID_CELL_BOXES

grid_columns=[]

def _create_output(d, img_width, img_height):
    cls = d['name']

    xmin = float(d['bndbox']['xmin'])
    ymin = float(d['bndbox']['ymin'])
    xmax = float(d['bndbox']['xmax'])
    ymax = float(d['bndbox']['ymax'])

    x, y, w, h = get_center_size(xmin, ymin, xmax, ymax, img_width, img_height)
    cls = to_categorical(class_encoder.transform([cls]), num_classes=len(classes))[0]

    output = np.concatenate(([x, y, w, h, 1], cls))
    return output


def _create_outputs(l, img_width, img_height):
    return [_create_output(d, img_width, img_height) for d in l][0:10]


def _preprocess_objects_dict(d, img_width, img_height):
    cls = d['name']

    xmin = float(d['bndbox']['xmin'])
    ymin = float(d['bndbox']['ymin'])
    xmax = float(d['bndbox']['xmax'])
    ymax = float(d['bndbox']['ymax'])

    x = (xmax + xmin) / 2
    y = (ymax + ymin) / 2

    cell_w = img_width // GRID_SIZE[0]
    cell_h = img_height // GRID_SIZE[1]

    anchor_x = x // cell_w
    anchor_y = y // cell_h

    x = (x - anchor_x*cell_w) / cell_w
    y =  (y - anchor_y*cell_h) / cell_h

    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    #x, y, w, h = get_center_size(xmin, ymin, xmax, ymax, img_width, img_height)
    cls = class_encoder.transform([cls])

    return {"grid_box": (anchor_x,anchor_y), "class": cls, "is_object": [1], "bb_coords": [x, y], "bb_sizes": [w, h], 'bb_area': w * h}


def _preprocess_objects_list(l, img_width, img_height):
    return [_preprocess_objects_dict(d, img_width, img_height) for d in l]

def find_largest(row):
    pass

def _create_output_grid(d):

    x = d['bb_coords'][0]
    y = d['bb_coords'][1]

    w = d['bb_sizes'][0]
    h = d['bb_sizes'][1]

    cls = to_categorical(d['class'], num_classes=len(classes))[0]

    output = np.concatenate(([x, y, w, h, 1], cls))
    return output

def _create_grid(row):
    grid_columns = []
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            objects = []
            for o in row['object']:
                #print(o)
                if o['grid_box'] == (i, j):
                     objects.append(_create_output_grid(o))
            if len(objects)==0:
                objects.append(np.zeros(5+len(classes)))
            row['grid_' + str(i) + '_' + str(j)] = pad_sequences([objects],maxlen=GRID_CELL_BOXES, dtype='float32', padding='post', truncating='post')
            grid_columns.append('grid_' + str(i) + '_' + str(j))
    return row

def preprocess(data):
    data.drop(['database', 'annotation', 'image', 'segmented'], axis=1, inplace=True)

    data['output'] = data.apply(
        lambda x: _create_outputs(x['object'], x['width'], x['height']), axis=1)
    data['output'] = data['output'].apply(lambda x: pad_sequences([x], SEQUENCE_LENGTH, dtype='float32', padding='post')).apply(np.vstack)

    data['object'] = data.apply(
        lambda x: _preprocess_objects_list(x['object'], x['width'], x['height']), axis=1)

    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes', 'bb_area', 'grid_box']:
        data[col] = data['object'].apply(lambda X: [[x[col] for x in X]])

    data['bb_index'] = data['bb_area'].apply(lambda x: np.argmax(x))

    data['class'] = data['class'].apply(lambda x: to_categorical(x, num_classes=len(classes)))

    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes']:
        data[col + '_single'] = data.apply(lambda row: np.asarray([row[col][0][row['bb_index']]]), axis=1)

    # for col in ['class', 'is_object', 'bb_coords', 'bb_sizes']:
    #     data[col] = data[col].apply(lambda x: pad_sequences(x, SEQUENCE_LENGTH, dtype='float32', padding='post'))
    # data[col] = data[col].apply(lambda X: [pad_sequences([x], SEQUENCE_LENGTH, dtype='float32', padding='post')for x in X])

    for col in ['class', 'is_object']:
        data[col] = data[col].apply(np.vstack)

    data['weights'] = data['is_object'].apply(lambda X: [X] * 2)
    # data['weights'] = data['is_object'].apply(np.hstack)

    for col in ['bb_coords', 'bb_sizes', 'bb_area']:
        data[col] = data[col].apply(np.hstack)
    # data[col] = data[col].apply(lambda X: [[x] for x in X])
    # for col in ['class','is_object', 'bb_coords', 'bb_sizes']:
    #    data[col] = data.apply(lambda row: np.asarray([row[col], row['is_object']]), axis=1)

    data = data.apply(_create_grid, axis=1)
    return data


def load_preprocess(filename):
    data = load_json(filename)
    return preprocess(data)

def load_preprocess_save(file_in, file_out):
    data = load_json(file_in)
    data = preprocess(data)
    data.to_pickle(file_out)
    return data

