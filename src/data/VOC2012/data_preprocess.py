from config import SEQUENCE_LENGTH
import numpy as np
from src.data.VOC2012.data import classes
from src.data.VOC2012.data_loader import load_json
from sklearn import preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences



le = preprocessing.LabelEncoder()


def _preprocess_objects_dict(d, img_width, img_height):
    cls = d['name']
    xmin = float(d['bndbox']['xmin']) / img_width
    ymin = float(d['bndbox']['ymin']) / img_height
    width = (float(d['bndbox']['xmax']) / img_width) - xmin
    height = (float(d['bndbox']['ymax']) / img_height) - ymin

    cls = le.transform([cls])

    return {"class": cls, "is_object": [1], "bb_coords": np.array([xmin, ymin]), "bb_sizes": np.array([width, height])}


def _preprocess_objects_list(l, img_width, img_height):
    le.fit(classes)
    return [_preprocess_objects_dict(d, img_width, img_height) for d in l][
           0:max(len(l), 10)]  # TODO change to multiple outputs later


# return [_preprocess_objects_dict(l[0], img_width, img_height)]


def preprocess(data):
    data.drop(['database', 'annotation', 'image', 'segmented'], axis=1, inplace=True)
    data['object'] = data.apply(
        lambda x: _preprocess_objects_list(x['object'], x['width'], x['height']), axis=1)

    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes']:
        data[col] = data['object'].apply(lambda X: [[x[col] for x in X]])
    for col in ['class', 'is_object']:
        data[col] = data[col].apply(lambda x: np.vstack(x))
    for col in ['class', 'is_object', 'bb_coords', 'bb_sizes']:
        data[col] = data[col].apply(lambda x: pad_sequences(x, SEQUENCE_LENGTH, dtype='float32', padding='post')[0])

    return data


def load_preprocess(filename):
    data = load_json(filename)
    return preprocess(data)
