from tensorflow import keras

BATCH_SIZE=32
CONV_BASE_SIZE = 8
#CONV_ACTIVATION ='relu'
CONV_ACTIVATION = keras.layers.LeakyReLU(0.1)
GRID_SIZE=(4,4)
GRID_CELL_BOXES=1
INPUT_SIZE=(256,256)
SEQUENCE_LENGTH = 5
VALIDATION_SPLIT=0.1

#YOLO_LAYERS_COUNTS = [1,2,2,1]
YOLO_LAYERS_COUNTS = [2,2,2,2]
#YOLO_LAYERS_COUNTS = [2,8,8,4]