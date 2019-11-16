from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import BATCH_SIZE
def datagen_flow(data):
    datagen = ImageDataGenerator()
    return datagen.flow_from_dataframe(data,
                                directory='../data/raw/VOC2012/JPEGImages',
                                x_col='filename',
                                y_col=['bb_coords', 'bb_sizes', 'class'],
                                target_size=(256, 256),
                                rescale = 1.0/255,
                                class_mode='multi_output',
                                batch_size=BATCH_SIZE)