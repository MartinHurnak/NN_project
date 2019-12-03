from tensorflow import keras
from tensorflow.keras.layers import Conv2D, add, GlobalAveragePooling2D, Dense, Concatenate, \
    Reshape, BatchNormalization
import shutil
from src.models.DataGen import DataGenGrid
from src.models.losses import SumSquaredLoss
from src.models.metrics import precision, recall
from tensorflow.keras import backend as K
from datetime import datetime
import os
import json


class YoloLayer(keras.layers.Layer):
    def __init__(self, filters_1, filters_2, activation='relu', batch_normalization=True, regularizer=None, **kwargs):
        super(YoloLayer, self).__init__(**kwargs)
        self.filters_1 = filters_1
        self.filters_2 = filters_2
        self.activation = activation
        self.conv_1 = Conv2D(filters_1, (1, 1), activation=activation, padding='same',
                             kernel_regularizer=regularizer, bias_regularizer=regularizer)
        self.conv_2 = Conv2D(filters_2, (3, 3), activation=activation, padding='same',
                             kernel_regularizer=regularizer, bias_regularizer=regularizer)
        self.batch_normalization = batch_normalization

    def call(self, x):
        l = self.conv_1(x)
        if self.batch_normalization:
            l = BatchNormalization()(l)
        l = self.conv_2(l)
        if self.batch_normalization:
            l = BatchNormalization()(l)
        l = add([x, l])
        return l

    def get_config(self):
        new_config = {}
        new_config.update({'filters_1': self.filters_1})
        new_config.update({'filters_2': self.filters_2})
        new_config.update({'activation': self.activation})
        base_config = super(YoloLayer, self).get_config()
        return dict(list(base_config.items()) + list(new_config.items()))


def create_model(config):
    input = keras.layers.Input(shape=(256, 256, 3,))
    model_layers = [
        Conv2D(config.CONV_BASE_SIZE, (3, 3), padding='same', activation=config.ACTIVATION,
               kernel_regularizer=config.REGULARIZER, bias_regularizer=config.REGULARIZER),
        Conv2D(2 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.ACTIVATION,
               kernel_regularizer=config.REGULARIZER, bias_regularizer=config.REGULARIZER),
        YoloLayer(config.CONV_BASE_SIZE, 2 * config.CONV_BASE_SIZE, activation=config.ACTIVATION,
                  batch_normalization=config.BATCH_NORMALIZATION),
        Conv2D(4 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.ACTIVATION,
               kernel_regularizer=config.REGULARIZER, bias_regularizer=config.REGULARIZER),

    ]
    model_layers += [YoloLayer(2 * config.CONV_BASE_SIZE, 4 * config.CONV_BASE_SIZE,
                               activation=config.ACTIVATION, batch_normalization=config.BATCH_NORMALIZATION,
                               regularizer=config.REGULARIZER)] * \
                    config.YOLO_LAYERS_COUNTS[0]
    model_layers.append(
        Conv2D(8 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.ACTIVATION,
               kernel_regularizer=config.REGULARIZER, bias_regularizer=config.REGULARIZER))
    model_layers += [YoloLayer(2 * config.CONV_BASE_SIZE, 8 * config.CONV_BASE_SIZE,
                               activation=config.ACTIVATION, batch_normalization=config.BATCH_NORMALIZATION,
                               regularizer=config.REGULARIZER)] * \
                    config.YOLO_LAYERS_COUNTS[1]
    model_layers.append(
        Conv2D(16 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.ACTIVATION,
               kernel_regularizer=config.REGULARIZER, bias_regularizer=config.REGULARIZER))
    model_layers += [YoloLayer(8 * config.CONV_BASE_SIZE, 16 * config.CONV_BASE_SIZE,
                               activation=config.ACTIVATION, batch_normalization=config.BATCH_NORMALIZATION,
                               regularizer=config.REGULARIZER)] * \
                    config.YOLO_LAYERS_COUNTS[2]
    model_layers.append(
        Conv2D(32 * config.CONV_BASE_SIZE, (3, 3), strides=2, padding='same', activation=config.ACTIVATION,
               kernel_regularizer=config.REGULARIZER, bias_regularizer=config.REGULARIZER))
    model_layers += [YoloLayer(16 * config.CONV_BASE_SIZE, 32 * config.CONV_BASE_SIZE,
                               activation=config.ACTIVATION, batch_normalization=config.BATCH_NORMALIZATION,
                               regularizer=config.REGULARIZER)] * \
                    config.YOLO_LAYERS_COUNTS[3]

    x = input
    for layer in model_layers:
        x = layer(x)
        if config.BATCH_NORMALIZATION:
            x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(config.DENSE_SIZE, activation=config.ACTIVATION, kernel_regularizer=config.REGULARIZER,
              bias_regularizer=config.REGULARIZER)(x)

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


def create_fit_evaluate(data, config, **kwargs):
    datagen = DataGenGrid(batch_size=config.BATCH_SIZE, input_size=(256, 256), validation_split=config.VALIDATION_SPLIT)

    K.clear_session()
    model = create_model(config)

    log = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs/TensorBoard", log),
            histogram_freq=1,
            update_freq='batch',
            profile_batch=0)
    ]
    print('Logs:', log)

    model.compile(loss=SumSquaredLoss(grid_size=config.GRID_SIZE, negative_box_coef=config.LOSS_NEGATIVE_BOX_COEF,
                                      size_coef=config.LOSS_SIZE_COEF,
                                      position_coef=config.LOSS_POSITION_COEF), metrics=[precision, recall],
                  optimizer=config.OPTIMIZER)

    history = model.fit_generator(datagen.flow_train(data),
                                  epochs=config.EPOCHS,
                                  validation_data=datagen.flow_val(data) if config.VALIDATION_SPLIT > 0.0 else None,
                                  callbacks=callbacks
                                  )
    val_metrics = {}
    if config.VALIDATION_SPLIT > 0:
        evaluation = model.evaluate_generator(datagen.flow_val(data))
        for k, v in zip(model.metrics_names, evaluation):
            val_metrics[k] = float(v)

    log_dict = {
        "log_name": log,
        "parameters": {
            "optimizer": config.OPTIMIZER.get_config(),
            "epochs": int(config.EPOCHS),
            "batch_size": int(config.BATCH_SIZE),
            "loss_koef_negative_box": float(config.LOSS_NEGATIVE_BOX_COEF),
            "loss_koef_position": float(config.LOSS_POSITION_COEF),
            "loss_koef_size_coef": float(config.LOSS_SIZE_COEF),
            "batch_normalization": bool(config.BATCH_NORMALIZATION),
            "regularization": config.REGULARIZER.get_config()
        },
        'val_metrics': val_metrics
    }

    script_dir = os.path.dirname(__file__)
    rel_path = "../../logs/log.json"
    path_json = os.path.join(script_dir, rel_path)

    with open(path_json, 'a+') as file_json:
        file_json.write(json.dumps(log_dict))
        file_json.write('\n')

    config_src = os.path.join(script_dir, '../../config.yaml')
    config_dst = os.path.join(os.path.join(script_dir, '../../logs/configs'), log + '.yaml')
    shutil.copy(config_src, config_dst)

    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_weights('models/{}.h5'.format(log))

    return model
