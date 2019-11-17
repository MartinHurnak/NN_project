from config import SEQUENCE_LENGTH
import numpy as np
from src.data.VOC2012.data import classes
from src.data.VOC2012.data_loader import load_json
from sklearn import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from src.helpers import get_center_size

class_encoder = preprocessing.LabelEncoder()


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

    x, y, w, h = get_center_size(xmin, ymin, xmax, ymax, img_width, img_height)
    cls = class_encoder.transform([cls])

    return {"class": cls, "is_object": [1], "bb_coords": [x, y], "bb_sizes": [w, h], 'bb_area': w * h}


def _preprocess_objects_list(l, img_width, img_height):
    return [_preprocess_objects_dict(d, img_width, img_height) for d in l][0:10]


# return [_preprocess_objects_dict(l[0], img_width, img_height)]


def preprocess(data):
    data.drop(['database', 'annotation', 'image', 'segmented'], axis=1, inplace=True)

    data['output'] = data.apply(
        lambda x: _create_outputs(x['object'], x['width'], x['height']), axis=1)
    data['output'] = data['output'].apply(lambda x: pad_sequences([x], SEQUENCE_LENGTH, dtype='float32', padding='post')).apply(np.vstack)

    data['object'] = data.apply(
        lambda x: _preprocess_objects_list(x['object'], x['width'], x['height']), axis=1)

    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes', 'bb_area']:
        data[col] = data['object'].apply(lambda X: [[x[col] for x in X]])

    data['bb_index'] = data['bb_area'].apply(lambda x: np.argmax(x))

    data['class'] = data['class'].apply(lambda x: to_categorical(x, num_classes=len(classes)))

    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes']:
        data[col + '_single'] = data.apply(lambda row: np.asarray([row[col][0][row['bb_index']]]), axis=1)

    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes']:
        data[col] = data[col].apply(lambda x: pad_sequences(x, SEQUENCE_LENGTH, dtype='float32', padding='post'))
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
    return data


def load_preprocess(filename):
    data = load_json(filename)
    return preprocess(data)

def load_preprocess_save(file_in, file_out):
    data = load_json(file_in)
    data = preprocess(data)
    data.to_pickle(file_out)
    return data

class_encoder.fit(classes)