from src.models.YOLOv3 import create_model
from src.models.DataGen import DataGenGrid
from src.models.losses import WholeOutputLoss
from config import GRID_SIZE
from tensorflow import keras
from tensorflow.keras import backend as K
from src.data.VOC2012.data import classes
from datetime import datetime

datagen = DataGenGrid(batch_size=32, input_size=(256,256), validation_split=0)

K.clear_session()
model = create_model(len(classes))

log = datetime.now().strftime('%Y_%m_%d_%H_%M_%s')
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir=os.path.join("logs", log),
        histogram_freq=1,
        profile_batch=0)
]
print('Logs:', log)
losses = [WholeOutputLoss(grid_x=x, grid_y=y, grid_size=GRID_SIZE, negative_box_coef=0.5) for x in range(GRID_SIZE[0]) for y in range(GRID_SIZE[1])]
#sgd= keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
#adam = keras.optimizers.Adam(clipvalue=0.5)
model.compile(loss=losses, optimizer='adam')
# model.fit_generator(datagen.flow_train(df.head(5)),
#                     epochs=10,
#                    # validation_data=datagen.flow_val(df.head(1))
#                     callbacks=callbacks
#                    )
model.summary()