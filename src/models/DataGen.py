from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import BATCH_SIZE, VALIDATION_SPLIT
from PIL import Image
import os
import numpy as np
from tensorflow import keras
from config import INPUT_SIZE
class DataGen:
    def __init__(self):
        self.datagen = ImageDataGenerator(validation_split=VALIDATION_SPLIT)

    def flow_train(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['bb_coords_single', 'bb_sizes_single', 'class_single'],
                                                target_size=INPUT_SIZE,
                                                rescale=1.0 / 255,
                                                class_mode='multi_output',
                                                batch_size=BATCH_SIZE,
                                                subset='training'
                                                )

    def flow_val(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['bb_coords_single', 'bb_sizes_single', 'class_single'],
                                                target_size=INPUT_SIZE,
                                                rescale=1.0 / 255,
                                                class_mode='multi_output',
                                                batch_size=BATCH_SIZE,
                                                subset='validation'
                                                )


class DataGenMultiOutput:
    def __init__(self):
        self.datagen = ImageDataGenerator(validation_split=VALIDATION_SPLIT)

    def flow_train(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['output'],
                                                #y_col=['bb_coords', 'bb_sizes', 'is_object', 'class'],
                                               # weight_col='weights',
                                                target_size=INPUT_SIZE,
                                                rescale=1.0 / 255,
                                                class_mode='multi_output',
                                                #class_mode='raw',
                                                batch_size=BATCH_SIZE,
                                                subset='training'
                                                )

    def flow_val(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['output'],
                                               # y_col=['bb_coords', 'bb_sizes', 'is_object', 'class'],
                                              #  weight_col='weights',
                                                target_size=INPUT_SIZE,
                                                rescale=1.0 / 255,
                                                class_mode='multi_output',
                                                #class_mode='raw',
                                                batch_size=BATCH_SIZE,
                                                subset='validation'
                                                )


def get_input(row):
    input = np.array(Image.open(os.path.join('data/raw/VOC2012/JPEGImages', row['filename'])).resize((256, 256)))
    return input


def get_output(row):
    output = []
    for col in ['is_object', 'class']:
        output.append(row[col])
    return output


def get_weights(row):
    weights = row['weights']
    return weights


class Generator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)

def generate_arrays_from_file(data, batch_size):
    while True:
        indices = np.arange(0, data.shape[0])
        np.random.shuffle(indices)

        while batch_size <= len(indices):
            batch = indices[0:batch_size]
            for i in range(batch_size):

                print(batch)
                indices = indices[batch_size:]
                inputs = np.array([get_input(row) for index, row in data.iloc[batch].iterrows()])
                outputs = [get_output(row) for index, row in data.iloc[batch].iterrows()]
                weights = [get_weights(row) for index, row in data.iloc[batch].iterrows()]
                print(inputs.shape)
                yield inputs, outputs, weights
