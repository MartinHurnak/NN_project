# Object Detection
Authors: Bc. Martin Hurňák, Bc. Kamil Mucha

## Motivation
Object detection is a main challenge of computer vision field. Thanks to the recent advancement in deep learning, we are able to train object detection applications, but they still require large datasets to achieve high levels of accuracy.

Object detection includes detection of individual objects on image, labeling them with correct class and creating a bounding box around them. Bounding box includes coordinates of the center of the object and height and width. We could also choose a different, more difficult approach to this, and go with segmentation. With segmentation we need to define exactly which pixels belong to the specific object. Since we are able to process images this way, we can also apply object detection on videos. With enough computing power and optimal algorithm we can even make it work in real time.

Object detection can be applied in many different fields. For example, tracking specific objects in the environment, video surveillance, pedestrian detection, anomaly detection, counting people and their movement and face detection.



## Related Work
If we aim to detect only one object, it can be formed as regression problem - our goal is to find foursome `(x, y, w, h)`,
 where `(x, y)` is position of bounding box, `w` is width and `h` is height of bounding box and label it with appropriate class.
  However, there can be more occurrences of object of interest in a picture, so we cannot solve this as simple 
  classification + regression problem as the length of output is variable. We need to find different regions of interest in an image
 and use CNN to classify this regions.

One of possible approaches for this task is called _Regions with CNN features_ or R-CNN. Selective search is used to
extract 2000 region proposals from given image. Regions are then processed by CNN which extracts feature vector. SVM
classifies presence of object in a region based on extracted features and bounding-box regressors predict offsets [1]. 
This approach is considered very slow as the CNN needs to process 2000 regions per image. In training phase, we also need 
to train its parts separately.

In _Spatial Pyramid Pooling network_ (SPP-net) image is fed into CNN that extracts feature maps 
only once. Region proposals are applied to feature maps. Feature maps are then processed by SPP-layer, which allows network 
to process images of arbitrary sizes. However, classification and regression is performed similarly to R-CNN, so 
architecture needs to be trained at multiple stages [2].

_Fast R-CNN_ method solves this problem by using single network architecture. It uses only single level SPP-layer called
RoI pooling layer which makes whole network trainable during single stage. It also introduces softmax layer for 
classification instead of SVM [3].

_Faster R-CNN_ further improves object detection process by replacing selective search with _Region Proposal Network_ (RPN)
that proposes regions from feature map. It serves as an 'attention' of network. Regions are then RoI pooled and classified
 as in Fast R-CNN. RPN can share convolutional layers with detection network (e.g. Fast R-CNN) [4]. This model is much 
 faster than original R-CNN, although it is still not enough fast if we want to detect objects in real-time 
 (e.g. in a video)

_You Only Look Once_ (YOLO) is extremely fast approach for object detection. Detection is reframed as a single regression
problem where bounding box coordinates and class probabilities are predicted straight from image pixels. Image is divided into 
_S x S_ grid, where each grid cell predicts _B_ bounding boxes and their confidence scores. Grid cell is responsible for 
detecting an object if its center falls into that grid cell. Non-max supression is then used to discard duplicate 
detections. YOLO  reasons globally about the image and is much better at generalizing in comparison to R-CNN methods [5].
However, detecting each of multiple smaller objects in a group (e.g. flock of birds or crowd of people in a distance) is a 
problem for YOLO as we can get only limited number of bounding box predictions per grid cell.

## Datasets
#### COCO, Common Objects in Context
This dataset offers images for object detection/segmentation training, validation and testing. For training there is 118K images available with captions of what is happening on the image and detection, classification and segmentation of the object on the image. For Validation there is 5K images available and 41K for testing. For classification there is 80 different classes ranging from toothbrushes, apples and oranges up to trains and boats.<br />
http://cocodataset.org/#home

#### VOC2012, Visual Object Classes Challenge 2012
This dataset offers 17125 images, from which all of them have annotations that describe what objects are on the picture and exact xmin, xmax, ymin, ymax coordinates where they begin and end. For classification there is 20 classes which focus on animals, vehicles, indoor objects and people.<br />
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html

## Solution Proposal
We would like to try method similar to YOLO algorithm, where we divide input image into a grid and predict multiple
bounding box coordinates relative to grid cell, because this approach should not take too long to train. We may 
experiment with different grid sizes and bounding box counts as their optimal value might be dependant on sizes and count 
of objects in images. 


## References
[1] Girshick, R., Donahue, J., Darrell, T. and Malik, J., 2014. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

[2] He, K., Zhang, X., Ren, S. and Sun, J., 2015. Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 37(9), pp.1904-1916.

[3] Girshick, R., 2015. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 1440-1448).

[4] Ren, S., He, K., Girshick, R. and Sun, J., 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems (pp. 91-99).

[5] Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
