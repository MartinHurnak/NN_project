import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, Activation, GlobalAveragePooling2D, Dense, concatenate, RepeatVector, \
    LSTM, Bidirectional, MaxPooling2D, Concatenate
from config import SEQUENCE_LENGTH, CONV_ACTIVATION, CONV_BASE_SIZE, LSTM_SIZE, YOLO_LAYERS_COUNTS, GRID_SIZE, GRID_CELL_BOXES

class YOLOv1Layer(keras.layers.Layer):
    def __init__(self, filters_1, filters_2, activation='relu'):
        super(YOLOv1Layer, self).__init__()
        self.conv_1 = Conv2D(filters_1, (1, 1), activation=activation, padding='same')
        self.conv_2 = Conv2D(filters_2, (3, 3), padding='same')

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


def create_model_multi_bb(num_classes):
    input = keras.layers.Input(shape=(448, 448, 3,))
    model_layers = [
        Conv2D(2*CONV_BASE_SIZE, (7, 7),strides=2, padding='same', activation=CONV_ACTIVATION),
        MaxPooling2D(padding='same'),
        Conv2D(6*CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION),
        MaxPooling2D(padding='same'),
        YOLOv1Layer(4*CONV_BASE_SIZE,8*CONV_BASE_SIZE),
        YOLOv1Layer(8*CONV_BASE_SIZE, 16*CONV_BASE_SIZE),
        MaxPooling2D(padding='same'),
        YOLOv1Layer(8*CONV_BASE_SIZE, 16*CONV_BASE_SIZE),
        YOLOv1Layer(8*CONV_BASE_SIZE, 16*CONV_BASE_SIZE),
        YOLOv1Layer(8*CONV_BASE_SIZE, 16*CONV_BASE_SIZE),
        YOLOv1Layer(8*CONV_BASE_SIZE, 16*CONV_BASE_SIZE),
        YOLOv1Layer(16*CONV_BASE_SIZE, 32*CONV_BASE_SIZE),
        MaxPooling2D(padding='same'),
        YOLOv1Layer(16*CONV_BASE_SIZE,  32*CONV_BASE_SIZE),
        YOLOv1Layer(16*CONV_BASE_SIZE,  32*CONV_BASE_SIZE),
        Conv2D( 32*CONV_BASE_SIZE, (3, 3), padding='same', activation=CONV_ACTIVATION),
        Conv2D( 32*CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=CONV_ACTIVATION),
        Conv2D( 32*CONV_BASE_SIZE, (3, 3), padding='same', activation=CONV_ACTIVATION),
        Conv2D( 32*CONV_BASE_SIZE, (3, 3), padding='same', activation=CONV_ACTIVATION),
    ]
    model_layers += [
        GlobalAveragePooling2D(),
        Dense(128*CONV_BASE_SIZE),
       # RepeatVector(SEQUENCE_LENGTH),
        #LSTM(32, return_sequences=True)
    ]

    x = input
    for layer in model_layers:
        x = layer(x)

    concats = []
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            for b in range(GRID_CELL_BOXES):
                bb_coord = Dense(2, activation='sigmoid', name='bb_coord_{}_{}_{}'.format(i,j,b))(x)
                bb_size = Dense(2, activation='sigmoid', name='bb_size_{}_{}_{}'.format(i,j,b))(x)
                has_object = Dense(1, activation='sigmoid', name='is_object_output_{}_{}_{}'.format(i,j,b))(x)
                classes = Dense(num_classes, activation='softmax', name='class_output_{}_{}_{}'.format(i,j,b))(x)
                concats.append(Concatenate(name='out_{}_{}_{}'.format(i,j,b))([bb_coord, bb_size, has_object, classes]))


    model = keras.Model(inputs=input, outputs=concats)

    return model
