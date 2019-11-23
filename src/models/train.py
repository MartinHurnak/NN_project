import pandas as pd
from src.models.YOLOv3 import create_and_fit

pickle_file = 'data/preprocessed/VOC2012/preprocessed.pkl'
df = pd.read_pickle(pickle_file)
df = df[df['has_person']].reset_index()
model = create_and_fit(df, 30, 256, verbose=1, val_split=0.1)
