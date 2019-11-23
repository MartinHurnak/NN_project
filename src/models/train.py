import pandas as pd
from src.models.YOLOv3 import create_and_fit
import config

pickle_file = 'data/preprocessed/VOC2012/preprocessed_train.pkl'
df = pd.read_pickle(pickle_file)
model = create_and_fit(df, config.EPOCHS, config.BATCH_SIZE, verbose=1, val_split=config.VALIDATION_SPLIT)
