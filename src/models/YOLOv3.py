from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, Activation, GlobalAveragePooling2D, Dense, concatenate, Concatenate, \
    BatchNormalization, GlobalMaxPooling2D, MaxPooling2D, Reshape
import config
from src.models.DataGen import DataGenGrid
from src.models.losses import SumSquaredLoss
from src.models.metrics import precision, recall
from tensorflow.keras import backend as K
from src.data.VOC2012.data import classes
from datetime import datetime
import os
import json


class YoloLayer(keras.layers.Layer):
    def __init__(self, filters_1, filters_2, activation=config.CONV_ACTIVATION, **kwargs):
        super(YoloLayer, self).__init__(**kwargs)
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.activation = activation
        self.conv_1 = Conv2D(filters_1, (1, 1), activation=activation, padding='same')
        self.conv_2 = Conv2D(filters_2, (3, 3), activation=activation, padding='same')

    def call(self, x):
        l = self.conv_1(x)
        l = self.conv_2(l)
        l = add([x, l])
        return l

    def get_config(self):
        new_config = {}
        new_config.update({'filters_1': self.filters_1})
        new_config.update({'filters_2': self.filters_2})
        new_config.update({'activation': self.activation})
        base_config = super(YoloLayer, self).get_config()
        return dict(list(base_config.items()) + list(new_config.items()))


def create_model(num_classes):
    input = keras.layers.Input(shape=(256, 256, 3,))
    model_layers = [
        Conv2D(config.CONV_BASE_SIZE, (3, 3), padding='same', activation=config.CONV_ACTIVATION),
        Conv2D(2 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.CONV_ACTIVATION),
        YoloLayer(config.CONV_BASE_SIZE, 2 * config.CONV_BASE_SIZE, activation=config.CONV_ACTIVATION),
        Conv2D(4 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.CONV_ACTIVATION),

    ]
    model_layers += [YoloLayer(2 * config.CONV_BASE_SIZE, 4 * config.CONV_BASE_SIZE,
                               activation=config.CONV_ACTIVATION)] * \
                    config.YOLO_LAYERS_COUNTS[0]
    model_layers.append(
        Conv2D(8 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.CONV_ACTIVATION))
    model_layers += [YoloLayer(2 * config.CONV_BASE_SIZE, 8 * config.CONV_BASE_SIZE,
                               activation=config.CONV_ACTIVATION)] * \
                    config.YOLO_LAYERS_COUNTS[1]
    model_layers.append(
        Conv2D(16 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.CONV_ACTIVATION))
    model_layers += [YoloLayer(8 * config.CONV_BASE_SIZE, 16 * config.CONV_BASE_SIZE,
                               activation=config.CONV_ACTIVATION)] * \
                    config.YOLO_LAYERS_COUNTS[2]
    model_layers.append(
        Conv2D(32 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.CONV_ACTIVATION))
    model_layers += [YoloLayer(16 * config.CONV_BASE_SIZE, 32 * config.CONV_BASE_SIZE,
                               activation=config.CONV_ACTIVATION)] * \
                    config.YOLO_LAYERS_COUNTS[3]
    model_layers += [
        GlobalAveragePooling2D(),
        Dense(4096, activation=config.CONV_ACTIVATION)
    ]

    x = input
    for layer in model_layers:
        x = layer(x)

    output = []
    for i in range(config.GRID_SIZE[0]):
        for j in range(config.GRID_SIZE[1]):
            for b in range(config.GRID_CELL_BOXES):
                bb_coord = Dense(2, activation='sigmoid', name='bb_coord_{}_{}_{}'.format(i, j, b))(x)
                bb_size = Dense(2, activation='sigmoid', name='bb_size_{}_{}_{}'.format(i, j, b))(x)
                has_object = Dense(1, activation='sigmoid', name='is_object_output_{}_{}_{}'.format(i, j, b))(x)
                # classes = Dense(num_classes, activation='softmax', name='class_output_{}_{}_{}'.format(i,j,b))(x)
                # concats.append(Concatenate(name='out_{}_{}_{}'.format(i,j,b))([bb_coord, bb_size, has_object, classes]))
                output.append(
                    Concatenate(name='out_{}_{}_{}'.format(i, j, b))([bb_coord, bb_size, has_object]))

    concat = Concatenate(name='output')(output)
    out = Reshape(target_shape=(-1, 16, 5))(concat)

    model = keras.Model(inputs=input, outputs=out)

    return model


def create_and_fit(data, epochs, batch_size, val_split=0.1, **kwargs):
    datagen = DataGenGrid(batch_size=batch_size, input_size=(256, 256), validation_split=val_split)

    K.clear_session()
    model = create_model(len(classes))

    log = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs/TensorBoard", log),
            histogram_freq=1,
            update_freq='batch',
            profile_batch=0)
    ]
    print('Logs:', log)

    model.compile(loss=SumSquaredLoss(negative_box_coef=kwargs['neg_box_coef'], size_coef=kwargs['size_coef'],
                                      position_coef=kwargs['position_coef']), metrics=[precision, recall],
                  optimizer='adam')

    history = model.fit_generator(datagen.flow_train(data),
                                  epochs=epochs,
                                  validation_data=datagen.flow_val(data) if val_split > 0.0 else None,
                                  callbacks=callbacks
                                  )

    log_dict = {
        "log_name": log,
        "parameters": {
            "learning_rate": K.eval(model.optimizer.lr),
            "epochs": epochs,
            "batch_size": batch_size,
            "loss_koef_negative_box": K.eval(model.loss.negative_box_coef),
            "loss_koef_position": K.eval(model.loss.position_coef),
            "loss_koef_size_coef": K.eval(model.loss.size_coef)
        }
    }
    log_json = json.dumps(str(log_dict))
    #print(log_json)
    
    script_dir = os.path.dirname(__file__)
    print(script_dir)
    rel_path = "../../logs/log.json"
    path_json = os.path.join(script_dir, rel_path)
    print(path_json)
    #path_json = '/NN_project/logs/log.json'
    
    with open(path_json) as file_json:
        json_log_file = json.load(file_json)
    
    json_log_file.append(log_json)
    
    with open(path_json, 'w') as file_json:
        json.dump(json_log_file, json_file)

    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_weights('models/{}.h5'.format(log))

    return model
