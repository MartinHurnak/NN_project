import os
import sys

os.chdir('/NN_project')
sys.path.append("/NN_project")

import pandas as pd
from src.models.YOLOv3 import create_fit_evaluate
import config
import argparse

parser = argparse.ArgumentParser(prog='Train')
parser.add_argument('--epochs', nargs='?', help='Number of epochs for fit', type=int, default=config.EPOCHS)
parser.add_argument('--batch_size', nargs='?', help='Batch size', type=int, default=config.BATCH_SIZE)
parser.add_argument('--val_split', nargs='?', help='Train/Validation split', type=int, default=config.VALIDATION_SPLIT)
parser.add_argument('--neg_box_coef', nargs='?', help='Constant for loss function', type=float,
                    default=config.LOSS_NEGATIVE_BOX_COEF)
parser.add_argument('--size_coef', nargs='?', help='Constant for loss function', type=float,
                    default=config.LOSS_SIZE_COEF)
parser.add_argument('--pos_coef', nargs='?', help='Constant for loss function', type=float,
                    default=config.LOSS_POSITION_COEF)
args = parser.parse_args()

pickle_file = 'data/preprocessed/VOC2012/preprocessed_train.pkl'
df = pd.read_pickle(pickle_file)
model = create_fit_evaluate(df, epochs=args.epochs, batch_size=args.batch_size, verbose=1, val_split=args.val_split,
                       neg_box_coef=args.neg_box_coef, position_coef=args.pos_coef, size_coef=args.size_coef)
