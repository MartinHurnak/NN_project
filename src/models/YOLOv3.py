import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, Activation, GlobalAveragePooling2D, Dense, concatenate, RepeatVector, \
    LSTM, Bidirectional
from config import SEQUENCE_LENGTH, CONV_ACTIVATION


class YoloLayer(keras.layers.Layer):
    def __init__(self, filters_1, filters_2, activation='relu'):
        super(YoloLayer, self).__init__()
        self.conv_1 = Conv2D(filters_1, (1, 1), activation=activation, padding='same')
        self.conv_2 = Conv2D(filters_2, (3, 3), padding='same')
        self.activation = Activation(activation)

    def call(self, x):
        l = self.conv_1(x)
        l = self.conv_2(l)
        l = add([x, l])
        l = self.activation(l)
        return l


class OutputLayer(keras.layers.Layer):
    def __init__(self, num_classes):
        super(OutputLayer, self).__init__()
        self.lstm = Bidirectional(LSTM(1024, return_sequences=True))

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

        return [classes, has_object, bb_coord, bb_size]


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
            # Dense(1000),
            RepeatVector(SEQUENCE_LENGTH),
            OutputLayer(num_classes)
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
