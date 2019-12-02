from tensorflow import keras
import yaml



class Config:
    GRID_SIZE = (4,4)
    GRID_CELL_BOXES = 1

    def __init__(self, config_file, **kwargs):
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
            self.BATCH_SIZE = kwargs['batch_size'] if 'batch_size' in kwargs and kwargs['batch_size'] else cfg['batch_size'] #64
            self.EPOCHS = kwargs['epochs'] if 'epochs' in kwargs and kwargs['epochs'] else cfg['epochs'] #30

            self.INPUT_SIZE = cfg['input_size']
            self.CONV_BASE_SIZE = kwargs['conv_base_size'] if 'conv_base_size' in kwargs and kwargs['conv_base_size'] else cfg['conv_base_size']
            self.DENSE_SIZE = kwargs['dense_size'] if 'dense_size' in kwargs and kwargs['dense_size'] else cfg['dense_size']
            self.ACTIVATION = keras.layers.LeakyReLU(0.1)
            self.LEARNING_RATE = kwargs['learning_rate'] if 'learning_rate' in kwargs and kwargs['learning_rate'] else cfg['learning_rate']
            self.OPTIMIZER = optimizers[cfg['optimizer']](learning_rate=self.LEARNING_RATE)
            self.L1 = kwargs['l1'] if 'l1' in kwargs and kwargs['l1'] else cfg['regularization']['l1']
            self.L2 = kwargs['l2'] if 'l2' in kwargs and kwargs['l2'] else cfg['regularization']['l2']
            self.REGULARIZER = keras.regularizers.l1_l2(l1=self.L1, l2=self.L2)
            self.BATCH_NORMALIZATION = kwargs['batch_normalization'] if 'batch_normalization' in kwargs and kwargs['batch_normalization'] else cfg[
                'batch_normalization']

            self.YOLO_LAYERS_COUNTS = kwargs['yolo_layers'] if 'yolo_layers' in kwargs and kwargs['yolo_layers'] else cfg['yolo_layers']

            self.LOSS_NEGATIVE_BOX_COEF = kwargs['neg_box_coef'] if 'neg_box_coef' in kwargs and kwargs['neg_box_coef'] else cfg['loss']['neg_box_coef']
            self.LOSS_POSITION_COEF = kwargs['position_coef'] if 'position_coef' in kwargs and kwargs['position_coef'] else cfg['loss']['position_coef']
            self.LOSS_SIZE_COEF = kwargs['size_coef'] if 'size_coef'in kwargs  and kwargs['size_coef'] else cfg['loss']['size_coef']

            self.TRAIN_TEST_SPLIT =  kwargs['train_test_split'] if 'train_test_split' in kwargs  and kwargs['train_test_split'] else cfg['train_test_split']
            self.VALIDATION_SPLIT = kwargs['validation_split'] if 'validation_split' in kwargs and kwargs['validation_split']  else cfg['validation_split']



            #self.SEQUENCE_LENGTH = 5


optimizers= {
    'adam': keras.optimizers.Adam,
    'sgd': keras.optimizers.SGD
}