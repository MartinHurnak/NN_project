from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import BATCH_SIZE, VALIDATION_SPLIT


class DataGen:
    def __init__(self):
        self.datagen = ImageDataGenerator(validation_split=VALIDATION_SPLIT)

    def flow_train(self, data):
        return self.datagen.flow_from_dataframe(data,
                                    directory='data/raw/VOC2012/JPEGImages',
                                    x_col='filename',
                                    y_col=['bb_coords', 'bb_sizes', 'class'],
                                    target_size=(256, 256),
                                    rescale = 1.0/255,
                                    class_mode='multi_output',
                                    batch_size=BATCH_SIZE,
                                    subset='training'
                                    )

    def flow_val(self, data):
        return self.datagen.flow_from_dataframe(data,
                                    directory='data/raw/VOC2012/JPEGImages',
                                    x_col='filename',
                                    y_col=['bb_coords', 'bb_sizes', 'class'],
                                    target_size=(256, 256),
                                    rescale = 1.0/255,
                                    class_mode='multi_output',
                                    batch_size=BATCH_SIZE,
                                    subset='validation'
                                )