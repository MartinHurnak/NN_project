# Object Detection

Authors: Bc. Martin Hurňák, Bc. Kamil Mucha

## Related Work
If we aim to detect only one object, it can be formed as regression problem - our goal is to find foursome `(x, y, w, h)`,
 where `(x, y)` is position of bounding box, `w` is width and `h` is height of bounding box. However, there can be more
 occurrences of object of interest in a picture, so we cannot solve this problem by simply combining convolutional and
 fully-connected layers as the length of output is variable. We need to find different regions of interest in an image
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
that proposes regions from feature map. **TODO** [4]

_YOLO_ (You Only Look Once) 
**TODO**



## References
[1] Girshick, R., Donahue, J., Darrell, T. and Malik, J., 2014. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

[2] He, K., Zhang, X., Ren, S. and Sun, J., 2015. Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 37(9), pp.1904-1916.

[3] Girshick, R., 2015. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 1440-1448).

[4] Ren, S., He, K., Girshick, R. and Sun, J., 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems (pp. 91-99).