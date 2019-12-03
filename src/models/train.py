import os
import sys

os.chdir('/NN_project')
sys.path.append("/NN_project")

import pandas as pd
from src.models.YOLOv3 import create_fit_evaluate
from src.config import Config
import argparse

parser = argparse.ArgumentParser(prog='Train')
parser.add_argument('--epochs', nargs='?', help='Number of epochs for fit', type=int)
parser.add_argument('--batch_size', nargs='?', help='Batch size', type=int)
parser.add_argument('--val_split', nargs='?', help='Train/Validation split', type=float)
parser.add_argument('--neg_box_coef', nargs='?', help='Constant for loss function', type=float)
parser.add_argument('--size_coef', nargs='?', help='Constant for loss function', type=float)
parser.add_argument('--pos_coef', nargs='?', help='Constant for loss function', type=float)
parser.add_argument('--lr', nargs='?', help='Learning rate', type=float)
parser.add_argument('--config', nargs='?', help='Config YAML file', type=str,
                    default='config.yaml')
args = parser.parse_args()


config = Config(args.config, epochs=args.epochs, batch_size=args.batch_size, val_split=args.val_split,
                       neg_box_coef=args.neg_box_coef, position_coef=args.pos_coef, size_coef=args.size_coef, learning_rate=args.lr)

pickle_file = 'data/preprocessed/VOC2012/preprocessed_train.pkl'
df = pd.read_pickle(pickle_file)
model = create_fit_evaluate(df, config, verbose=1)
