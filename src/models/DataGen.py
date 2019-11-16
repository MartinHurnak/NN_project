from tensorflow.keras.preprocessing.image import ImageDataGenerator

def datagen_flow(data):
    datagen = ImageDataGenerator()
    return datagen.flow_from_dataframe(data,
                                directory='../data/raw/VOC2012/JPEGImages',
                                x_col='filename',
                                y_col=['class', 'is_object', 'bb_coords', 'bb_sizes'],
                                target_size=(256, 256),
                                class_mode='multi_output',
                                batch_size=32)