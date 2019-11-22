from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, Activation, GlobalAveragePooling2D, Dense, concatenate, Concatenate, \
    BatchNormalization, GlobalMaxPooling2D,MaxPooling2D, Reshape
from config import CONV_ACTIVATION, CONV_BASE_SIZE, YOLO_LAYERS_COUNTS, GRID_SIZE, GRID_CELL_BOXES
from src.models.DataGen import DataGenGrid
from src.models.losses import SumSquaredLoss
from src.models.metrics import precision
from tensorflow.keras import backend as K
from src.data.VOC2012.data import classes
from datetime import datetime
import os

class YoloLayer(keras.layers.Layer):
    def __init__(self, filters_1, filters_2, activation=CONV_ACTIVATION):
        super(YoloLayer, self).__init__()
        self.conv_1 = Conv2D(filters_1, (1, 1), activation=activation, padding='same')
        self.conv_2 = Conv2D(filters_2, (3, 3), activation=activation, padding='same')

    def call(self, x):
        l = self.conv_1(x)
        l = self.conv_2(l)
        l = add([x, l])
        return l


def create_model(num_classes):
    input = keras.layers.Input(shape=(256, 256, 3,))
    model_layers = [
        Conv2D(CONV_BASE_SIZE, (3, 3), padding='same', activation=CONV_ACTIVATION),
        Conv2D(2 * CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION),
        YoloLayer(CONV_BASE_SIZE, 2 * CONV_BASE_SIZE, activation=CONV_ACTIVATION),
        Conv2D(4 * CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION),

    ]
    model_layers += [YoloLayer(2 * CONV_BASE_SIZE, 4 * CONV_BASE_SIZE, activation=CONV_ACTIVATION)] * \
                    YOLO_LAYERS_COUNTS[0]
    model_layers.append(Conv2D(8 * CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
    model_layers += [YoloLayer(2 * CONV_BASE_SIZE, 8 * CONV_BASE_SIZE, activation=CONV_ACTIVATION)] * \
                    YOLO_LAYERS_COUNTS[1]
    model_layers.append(Conv2D(16 * CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
    model_layers += [YoloLayer(8 * CONV_BASE_SIZE, 16 * CONV_BASE_SIZE, activation=CONV_ACTIVATION)] * \
                    YOLO_LAYERS_COUNTS[2]
    model_layers.append(Conv2D(32 * CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
    model_layers += [YoloLayer(16 * CONV_BASE_SIZE, 32 * CONV_BASE_SIZE, activation=CONV_ACTIVATION)] * \
                    YOLO_LAYERS_COUNTS[3]
    model_layers += [
        GlobalAveragePooling2D(),
        Dense(4096, activation=CONV_ACTIVATION)
    ]

    x = input
    for layer in model_layers:
        x = layer(x)

    output = []
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            for b in range(GRID_CELL_BOXES):
                bb_coord = Dense(2, activation='sigmoid', name='bb_coord_{}_{}_{}'.format(i, j, b))(x)
                bb_size = Dense(2, activation='sigmoid', name='bb_size_{}_{}_{}'.format(i, j, b))(x)
                has_object = Dense(1, activation='sigmoid', name='is_object_output_{}_{}_{}'.format(i, j, b))(x)
                # classes = Dense(num_classes, activation='softmax', name='class_output_{}_{}_{}'.format(i,j,b))(x)
                # concats.append(Concatenate(name='out_{}_{}_{}'.format(i,j,b))([bb_coord, bb_size, has_object, classes]))
                output.append(
                    Concatenate(name='out_{}_{}_{}'.format(i, j, b))([bb_coord, bb_size, has_object]))

    concat = Concatenate(name='output')(output)
    out=Reshape(target_shape=(-1,16,5))(concat)

    model = keras.Model(inputs=input, outputs=out)

    return model

def create_and_fit(data, epochs, batch_size, val_split=0.1, **kwargs):
    datagen = DataGenGrid(batch_size=batch_size, input_size=(256,256), validation_split=val_split)

    K.clear_session()
    model = create_model(len(classes))

    log = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", log),
            histogram_freq=1,
            update_freq='batch',
            profile_batch=0)
    ]
    print('Logs:', log)

    model.compile(loss=SumSquaredLoss(negative_box_coef=0.25), metrics=[precision], optimizer='adam')
    model.fit_generator(datagen.flow_train(data),
                        epochs=epochs,
                        validation_data=datagen.flow_val(data),
                        callbacks=callbacks,
                        **kwargs
                       )
    return model