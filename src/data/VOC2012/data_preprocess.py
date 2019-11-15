from src.data.VOC2012.data_loader import load_json
import numpy as np
from sklearn import preprocessing
from src.data.VOC2012.data import classes

le = preprocessing.LabelEncoder()


def _preprocess_objects_dict(d, img_width, img_height):
    cls = d['name']
    xmin = float(d['bndbox']['xmin']) / img_width
    ymin = float(d['bndbox']['ymin']) / img_height
    width = (float(d['bndbox']['xmax']) / img_width) - xmin
    height = (float(d['bndbox']['ymax']) / img_height) - ymin

    bndbox = np.array([xmin, ymin, width, height])
    cls = le.transform([cls])

    return {"class": cls, "object": 1, "bb": bndbox}


def _preprocess_objects_list(l, img_width, img_height):
    le.fit(classes)
    # return [_preprocess_objects_dict(d, img_width, img_height) for d in l]
    return [_preprocess_objects_dict(l[0], img_width, img_height)]  # TODO change to multiple outputs later


def preprocess(data):
    data.drop(['database', 'annotation', 'image', 'segmented'], axis=1, inplace =True)
    data['object'] = data.apply(
        lambda x: _preprocess_objects_list(x['object'], x['width'], x['height']), axis=1)

    return data


def load_preprocess(filename):
    data = load_json(filename)
    return preprocess(data)
