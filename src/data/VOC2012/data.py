from sklearn import preprocessing
from config import GRID_SIZE

classes = ['aeroplane',
           'bicycle',
           'bird',
           'boat',
           'bottle',
           'bus',
           'car',
           'cat',
           'chair',
           'cow',
           'diningtable',
           'dog',
           'horse',
           'motorbike',
           'person',
           'pottedplant',
           'sheep',
           'sofa',
           'train',
           'tvmonitor']

class_encoder = preprocessing.LabelEncoder()
class_encoder.fit(classes)

grid_columns = []
for i in range(GRID_SIZE[0]):
    for j in range(GRID_SIZE[1]):
        grid_columns.append('grid_' + str(i) + '_' + str(j))
