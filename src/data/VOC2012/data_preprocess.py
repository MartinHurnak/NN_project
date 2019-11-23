import numpy as np
from src.data.VOC2012.data import classes, class_encoder, grid_columns
from src.data.VOC2012.data_loader import load_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from src.helpers import get_center_size
from sklearn.model_selection import train_test_split
import config

def _create_output(d, img_width, img_height):
    cls = d['name']
    # if not cls=='person':
    #     return None
    xmin = float(d['bndbox']['xmin'])
    ymin = float(d['bndbox']['ymin'])
    xmax = float(d['bndbox']['xmax'])
    ymax = float(d['bndbox']['ymax'])

    x, y, w, h = get_center_size(xmin, ymin, xmax, ymax, img_width, img_height)
    cls = to_categorical(class_encoder.transform([cls]), num_classes=len(classes))[0]

    output = np.concatenate(([x, y, w, h, 1], cls))
    return output


def _create_outputs(l, img_width, img_height):
    return [_create_output(d, img_width, img_height) for d in l if d['name']=='person'][0:10]


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

    cls = class_encoder.transform([cls])

    return {"grid_box": (anchor_x,anchor_y), "class": cls, "is_object": [1], "bb_coords": [x, y], "bb_sizes": [w, h], 'bb_area': w * h}


def _preprocess_objects_list(l, img_width, img_height):
    return [_preprocess_objects_dict(d, img_width, img_height) for d in l]


def _create_output_grid(d):
    x = d['bb_coords'][0]
    y = d['bb_coords'][1]

    w = d['bb_sizes'][0]
    h = d['bb_sizes'][1]

    cls = to_categorical(d['class'], num_classes=len(classes))[0]
    output = [x, y, w, h, 1]
    #output = np.concatenate(([x, y, w, h, 1], cls))
    return output

def _create_grid(row):

    for i in range(config.GRID_SIZE[0]):
        for j in range(config.GRID_SIZE[1]):
            objects = []
            for o in row['object']:
                #print(o)
                if o['grid_box'] == (i, j) and o['class']==14:
                     objects.append(_create_output_grid(o))
            if len(objects)==0:
                objects.append(np.zeros(5))
                #objects.append(np.zeros(5+len(classes)))
            row['grid_' + str(i) + '_' + str(j)] = pad_sequences([objects],maxlen=config.GRID_CELL_BOXES, dtype='float32', padding='post', truncating='post')
    return row

def preprocess(data):
    data.drop(['database', 'annotation', 'image', 'segmented'], axis=1, inplace=True)

    data['output'] = data.apply(
        lambda x: _create_outputs(x['object'], x['width'], x['height']), axis=1)

    data['output'] = data['output'].apply(lambda x: pad_sequences([x], config.SEQUENCE_LENGTH, dtype='float32', padding='post')).apply(np.vstack)

    data['has_person'] = data['object'].apply(lambda D: np.array([(d['name'] == 'person') for d in D]).any())

    data['object'] = data.apply(
        lambda x: _preprocess_objects_list(x['object'], x['width'], x['height']), axis=1)

    data = data.apply(_create_grid, axis=1)

    data['grid_output']=data[grid_columns].apply(np.hstack, axis=1)
    data = data[data['has_person']].reset_index()
    return data

def split_train_test(data, test_ratio, file_out_train, file_out_test):
    train, test = train_test_split(np.array(data.index), test_size=test_ratio)
    data_train = data.iloc[train]
    data_test = data.iloc[test]
    data_train.to_pickle(file_out_train)
    data_test.to_pickle(file_out_test)
    return data_train, data_test

def load_preprocess(filename):
    data = load_json(filename)
    return preprocess(data)

def load_preprocess_save(file_in, file_out_train, file_out_test):
    data = load_json(file_in)
    data = preprocess(data)
    data_train, data_test = split_train_test(data, config.TRAIN_TEST_SPLIT, file_out_train, file_out_test)
    return data_train, data_test

