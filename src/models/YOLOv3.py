import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, Activation, GlobalAveragePooling2D, Dense, concatenate


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
        self.bb = Dense(4, activation='sigmoid', name='bb_output')
        self.has_object = Dense(1, activation='sigmoid', name='is_object_output')
        self.classes = Dense(num_classes, activation='softmax', name='class_output')

    def call(self, x):
        bb = self.bb(x)
        has_object = self.has_object(x)
        classes = self.classes(x)

        return [classes, has_object, bb]


class YOLOv3(keras.Model):
    '''
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
    '''

    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        self.model_layers = [
            Conv2D(32, (3, 3), padding='same'),
            Conv2D(64, (3, 3), strides=2, padding='same'),
            YoloLayer(32, 64),
            Conv2D(128, (3, 3), strides=2, padding='same'),
        ]
        self.model_layers += [YoloLayer(64, 128)]   *2
        self.model_layers.append(Conv2D(256, (3, 3), strides=2, padding='same'))
        self.model_layers += [YoloLayer(128, 256)]  # *8
        self.model_layers.append(Conv2D(512, (3, 3), strides=2, padding='same'))
        self.model_layers += [YoloLayer(256, 512)]  # *8
        self.model_layers.append(Conv2D(1024, (3, 3), strides=2, padding='same'))
        self.model_layers += [YoloLayer(512, 1024)]  # *4
        self.model_layers += [
            GlobalAveragePooling2D(),
            Dense(1000),
            OutputLayer(num_classes)
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
