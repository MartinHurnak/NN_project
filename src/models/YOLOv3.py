import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, Activation, GlobalAveragePooling2D, Dense, concatenate, RepeatVector, \
    LSTM, Bidirectional
from config import SEQUENCE_LENGTH, CONV_ACTIVATION, CONV_BASE_SIZE, LSTM_SIZE, YOLO_LAYERS_COUNTS
from src.models import losses


class YoloLayer(keras.layers.Layer):
    def __init__(self, filters_1, filters_2, activation='relu'):
        super(YoloLayer, self).__init__()
        self.conv_1 = Conv2D(filters_1, (1, 1), activation=activation, padding='same')
        self.conv_2 = Conv2D(filters_2, (3, 3), padding='same')
       # self.activation = Activation(activation)

    def call(self, x):
        l = self.conv_1(x)
        l = self.conv_2(l)
        l = add([x, l])
        #l = self.activation(l)
        return l


class OutputLayer(keras.layers.Layer):
    def __init__(self, num_classes):
        super(OutputLayer, self).__init__()
        self.bb_coord = Dense(2, activation='sigmoid', name='bb_coord')
        self.bb_size = Dense(2, activation='sigmoid', name='bb_size')
        self.has_object = Dense(1, activation='sigmoid', name='is_object_output')
        self.classes = Dense(num_classes, activation='softmax', name='class_output')

    def call(self, x):
        x = self.lstm(x)
        bb_coord = self.bb_coord(x)
        bb_size = self.bb_size(x)
        has_object = self.has_object(x)
        classes = self.classes(x)

        return has_object, bb_coord, bb_size, classes


class YOLOv3(keras.Model):
    '''
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
    '''

    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.model_layers = [
            Conv2D(32, (3, 3), padding='same', activation=CONV_ACTIVATION),
            Conv2D(64, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION),
            YoloLayer(32, 64, activation=CONV_ACTIVATION),
            Conv2D(128, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION),
        ]
        self.model_layers += [YoloLayer(64, 128, activation=CONV_ACTIVATION)] * 2
        self.model_layers.append(Conv2D(256, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
        self.model_layers += [YoloLayer(128, 256, activation=CONV_ACTIVATION)]  # *8
        self.model_layers.append(Conv2D(512, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
        self.model_layers += [YoloLayer(256, 512, activation=CONV_ACTIVATION)]  # *8
        self.model_layers.append(Conv2D(1024, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
        self.model_layers += [YoloLayer(512, 1024, activation=CONV_ACTIVATION)]  # *4
        self.model_layers += [
            GlobalAveragePooling2D(),
            Dense(1000),
            # RepeatVector(SEQUENCE_LENGTH),
            # OutputLayer(num_classes)
        ]
        self.output_layers = [
            Dense(2, activation='sigmoid', name='bb_coord'),
            Dense(2, activation='sigmoid', name='bb_size'),
            Dense(num_classes, activation='softmax', name='class_output')
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)

        return [layer(x) for layer in self.output_layers]


def create_model_single_bb(num_classes):
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
    model_layers += [YoloLayer(8 * CONV_BASE_SIZE, 16 * CONV_BASE_SIZE,
                               activation=CONV_ACTIVATION)] * YOLO_LAYERS_COUNTS[2]
    model_layers.append(Conv2D(32 * CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION))
    model_layers += [YoloLayer(16 * CONV_BASE_SIZE, 32 * CONV_BASE_SIZE,
                               activation=CONV_ACTIVATION)] * YOLO_LAYERS_COUNTS[3]

    model_layers += [
        GlobalAveragePooling2D(),
        Dense(1000),
    ]
    x = input
    for layer in model_layers:
        x = layer(x)
    bb_coord = Dense(2, activation='tanh', name='bb_coord')(x)
    bb_size = Dense(2, activation='sigmoid', name='bb_size')(x)
    classes = Dense(num_classes, activation='softmax', name='class_output')(x)
    model = keras.Model(inputs=input, outputs=[bb_coord, bb_size, classes])
    return model


def create_model_multi_bb(num_classes):
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
        RepeatVector(SEQUENCE_LENGTH),
        LSTM(LSTM_SIZE, return_sequences=True)
    ]

    x = input
    for layer in model_layers:
        x = layer(x)
    bb_coord = Dense(2, activation='sigmoid', name='bb_coord')(x)
    bb_size = Dense(2, activation='sigmoid', name='bb_size')(x)
    has_object = Dense(1, activation='sigmoid', name='is_object_output')(x)
    classes = Dense(num_classes, activation='softmax', name='class_output')(x)

    con = concatenate([bb_coord, bb_size, has_object, classes])
    model = keras.Model(inputs=input, outputs=con)

    return model
