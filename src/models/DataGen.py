from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import BATCH_SIZE, VALIDATION_SPLIT
from config import INPUT_SIZE
from src.data.VOC2012.data import grid_columns


class DataGenGrid:
    def __init__(self, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, validation_split=VALIDATION_SPLIT):
        self.datagen = ImageDataGenerator(validation_split=validation_split, rescale=1.0 / 255.0)
        self.batch_size = batch_size
        self.input_size = input_size

    def flow_train(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['grid_output'],
                                                target_size=self.input_size,
                                                class_mode='multi_output',
                                                batch_size=self.batch_size,
                                                subset='training'
                                                )

    def flow_val(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['grid_output'],
                                                target_size=self.input_size,
                                                class_mode='multi_output',
                                                batch_size=self.batch_size,
                                                subset='validation',
                                                shuffle=False
                                                )

    def flow_test(self, data):
        return self.datagen.flow_from_dataframe(data,
                                                directory='data/raw/VOC2012/JPEGImages',
                                                x_col='filename',
                                                y_col=['grid_output'],
                                                target_size=self.input_size,
                                                class_mode='multi_output',
                                                batch_size=self.batch_size,
                                                subset=None,
                                                shuffle=False
                                                )
