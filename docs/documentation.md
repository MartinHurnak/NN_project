# Object Detection
Authors: Bc. Martin Hurňák, Bc. Kamil Mucha

## Motivation
Object detection is a main challenge of computer vision field. Thanks to the recent advancement in deep learning, we are able to train object detection applications, but they still require large datasets to achieve high levels of accuracy.

Object detection includes detection of individual objects on image, labeling them with correct class and creating a bounding box around them. Bounding box includes coordinates of the center of the object and height and width. We could also choose a different, more difficult approach to this, and go with segmentation. With segmentation we need to define exactly which pixels belong to the specific object. Since we are able to process images this way, we can also apply object detection on videos. With enough computing power and optimal algorithm we can even make it work in real time.

Object detection can be applied in many different fields. For example, tracking specific objects in the environment, video surveillance, pedestrian detection, anomaly detection, counting people and their movement and face detection.


## Related Work
If we aim to detect only one object, it can be formed as regression problem - our goal is to find foursome `(x, y, w, h)`, where `(x, y)` is position of bounding box, `w` is width and `h` is height of bounding box and label it with appropriate class. However, there can be more occurrences of object of interest in a picture, so we cannot solve this as simple classification + regression problem as the length of output is variable. We need to find different regions of interest in an image and use CNN to classify this regions.

#### R-CNN
One of possible approaches for this task is called _Regions with CNN features_ or R-CNN. Selective search is used to extract 2000 region proposals from given image. Regions are then processed by CNN which extracts feature vector. SVM classifies presence of object in a region based on extracted features and bounding-box regressors predict offsets [1]. This approach is considered very slow as the CNN needs to process 2000 regions per image. In training phase, we also need to train its parts separately.

#### SPP-net
In _Spatial Pyramid Pooling network_ (SPP-net) image is fed into CNN that extracts feature maps only once. Region proposals are applied to feature maps. Feature maps are then processed by SPP-layer, which allows network to process images of arbitrary sizes. However, classification and regression is performed similarly to R-CNN, so architecture needs to be trained at multiple stages [2].

#### Fast R-CNN
_Fast R-CNN_ method solves this problem by using single network architecture. It uses only single level SPP-layer called RoI pooling layer which makes whole network trainable during single stage. It also introduces softmax layer for classification instead of SVM [3].

#### Faster R-CNN
_Faster R-CNN_ further improves object detection process by replacing selective search with _Region Proposal Network_ (RPN) that proposes regions from feature map. It serves as an 'attention' of network. Regions are then RoI pooled and classified as in Fast R-CNN. RPN can share convolutional layers with detection network (e.g. Fast R-CNN) [4]. This model is much faster than original R-CNN, although it is still not enough fast if we want to detect objects in real-time (e.g. in a video)

#### YOLO
_You Only Look Once_ (YOLO) is extremely fast approach for object detection. Detection is reframed as a single regression problem where bounding box coordinates and class probabilities are predicted straight from image pixels. Image is divided into _S x S_ grid, where each grid cell predicts _B_ bounding boxes and their confidence scores. Grid cell is responsible for detecting an object if its center falls into that grid cell. Non-max supression is then used to discard duplicate detections. YOLO  reasons globally about the image and is much better at generalizing in comparison to R-CNN methods [5]. However, detecting each of multiple smaller objects in a group (e.g. flock of birds or crowd of people in a distance) is a problem for YOLO as we can get only limited number of bounding box predictions per grid cell.


## Datasets
#### COCO, Common Objects in Context
This dataset offers images for object detection/segmentation training, validation and testing. For training there is 118K images available with captions of what is happening on the image and detection, classification and segmentation of the object on the image. For Validation there is 5K images available and 41K for testing. For classification there is 80 different classes ranging from toothbrushes, apples and oranges up to trains and boats.<br />
http://cocodataset.org/#home

#### VOC2012, Visual Object Classes Challenge 2012
This dataset offers 17125 images, from which all of them have annotations that describe what objects are on the picture and exact xmin, xmax, ymin, ymax coordinates where they begin and end. For classification there is 20 classes which focus on animals, vehicles, indoor objects and people.<br />
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html


## Model
We have experimented with approach similar to YOLO algorithm, and we have focused on detecting only one specific class of objects. Similar to YOLO algorithm, we have divided images into S x S grid and predicted one bounding box for each grid cell. For feature extraction we have used architecture from YOLOv3 paper [6], but we have experimented with different layer sizes and convolutional-convolutional-residual layers count. As activation we have used Leaky ReLU (same as in paper).

![architerture of YOLOv3 model](images/YOLOv3.jpg "YOLOv3 architeture [6]")

#### Outputs
Output of our model consists of outputs for each grid cell. One bounding box is defined by tuple `(x, y, w, h, c)` where
`(x, y)` is position offset from top left corner of bounding box relative to grid cell, `(w, h)` is size of bounding box relative to whole image and `c` is confidence of prediction. This method was inspired by original YOLO paper [5]. All of the values mentioned are from <0,1> interval, so our output layer consists of `S x S x 5` neurons with sigmoid activation function. We have trained our network for detecting only one specific class (e.g. people). Our model then looks like this:

![](images/our_model_h.png)

#### Loss function
As loss function, we use sum squared error proposed by [5], where we omit box classification part. Our modified loss function 
then looks like this:

![original YOLO loss with omitted box classification par](images/loss.jpg "part of original YOLO sum squared loss [5]")

Ones with obj/noobj mean that part is only calculated for part that do (1<sup>obj</sup>) or do not (1<sup>noobj</sup>) contain objects. Lambdas 
in loss function are constant coeficients to prevent gradient from cells that do not contain object overpower
gradient from cells that do. Original paper uses &lambda;<sub>noobj</sub> = 0.5 and &lambda;<sub>coef</sub> = 5. Probably,
we will experiment with &lambda;<sub>noobj</sub> as it seems to be important hyperparameter for precision-recall ratio.


## Training <!--Description of the training routine.-->
For training of our model, we have decided to focus on detecting people, as this class is most common object in our explored image datasets. For training we have also used the opportunity to work with google cloud compute engine.

#### Used dataset and preprocessing
We have decided to use VOC2012 dataset, which has 9583 images containing at least one person. This dataset also required slight preprocessing changes to the way bounding box are described. As described in outputs of our model, our model represents bounding boxes by center offset from the top left corner of grid box and by width/height of bounding box, where this dataset had exact xmin, xmax, ymin, ymax coordinates where the bounding box begins and ends.

#### Model configuration
We have setup a config.yaml file for configuration of our model. Here we can easily adjust batch size, amount of epochs, image input size, convolution layers size, dense layer size, number of "YOLO" layers (1x1 convolution, 3x3 convolution, residual connection), enable/disable batch normalization, change used optimizer (from SGD and Adam options), set initial learning rate, enable/disable learning rate scheduling, adjust loss coefficient of negative box, position and size and adjust regularization.

## Experiments <!--Description of the experiments you conducted.-->

### Hyperparameters
In our experiments, we have optimized following hyperparameters:
  - Learning rate - this was probably the most important hyperparameter. Although we used Adam optimizer most of the time, initial learning rate is important as we found that bigger model is highly unstable in first few epochs, resulting in NaN loss if the learning rate is too high. Learning rate needs to be low at the beginning, while we can increase it later during training. For this we used learning rate scheduler. If scheduler is enabled we double the learning rate at the end of 5th, 10th and 15th epochs. If training takes longer than 30 epochs, we reduce learning rate by half at the end of 30th, 35th and 40th epochs as well.
  - Negative box loss coeficient (&lambda;<sub>noobj</sub>) - as described in Loss part, this coeficient reduces penalization of model for predicting boxes, where they should not be, to prevent situation where model does not predict any boxes, as the gradients from cells that do not contain object can easily overpower gradient from cells that do contain an object. We have tried &lambda;<sub>noobj</sub> from interval <0.03, 0.5>. Best precision-recall ratios are achieved with &lambda;<sub>noobj</sub> somewhere around 0.05
  - Convolutional layer sizes - we manually adjusted size of first convolutional layer only, sizes of other layers are adjusted automatically according to first layer to preserve ratios from original YOLO v3 extractor. We started with 4 filters at first layer and slowly increased it up to 32 filters at first layer (which results in 1024 filters in last convolutional layer).
  - Number of "YOLO layers" (one 1x1 convolution, 3x3 convolution, residual connection block). Original YOLO v3 feature extractor uses five layers consisting of [1, 2, 8, 8, 4] block respectively we started at [1, 2, 2, 2, 2] and increased counts later on. At [1, 2, 8, 8, 4] model was highly unstable during first epochs, which we could not solve even by decreasing learning rate. Best results were achieved using [1, 2, 4, 4, 4] model.
 - Batch normalization - we tried batch normalization as well, although we stopped using it as it caused model to overfit training data heavily while not decreasing validation loss at all.
 - Regularization - to prevent overfitting we used mainly L2 regularization of weights and biases (although we tried L1L2 regularization as well). L2 regularization in our experiments is usually set between <0.0001, 0.0003> with best results at 0.0002.

#### Logging
We were logging all input hyperparameters from config file and for validation metrics we were logging loss, precision and recall. All logs were saved in JSON format in log.json file with first pair of information being the `log name : timestamp`. As addition to this, we have aswell used the tensorboard logging function and we have kept the logs under the same name as in log.json.

#### Log table of trainings
<table class="table table-bordered table-hover table-condensed">
<thead><tr><th title="Field #1">log_name</th>
<th title="Field #2">optimizer</th>
<th title="Field #3">learning_rate</th>
<th title="Field #4">epochs</th>
<th title="Field #5">batch_size</th>
<th title="Field #6">loss_koef_negative_box</th>
<th title="Field #7">loss_koef_position</th>
<th title="Field #8">loss_koef_size_coef</th>
<th title="Field #9">batch_normalization</th>
<th title="Field #10">regularization/l1</th>
<th title="Field #11">regularization/l2</th>
<th title="Field #12">loss</th>
<th title="Field #13">precision</th>
<th title="Field #14">recall</th>
</tr></thead>
<tbody><tr>
<td>2019-11-30-09-42-27</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.15</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.7316289033208574</td>
<td align="right">0.41366973519325256</td>
<td align="right">0.35474807024002075</td>
</tr>
<tr>
<td>2019-11-30-10-23-58</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.2</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.7051303812435694</td>
<td align="right">0.5051430463790894</td>
<td align="right">0.29482102394104004</td>
</tr>
<tr>
<td>2019-11-30-11-05-08</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.25</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.8116266642298018</td>
<td align="right">0.5756961107254028</td>
<td align="right">0.26377609372138977</td>
</tr>
<tr>
<td>2019-11-30-11-46-23</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.3</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.7842261791229248</td>
<td align="right">0.5929040908813477</td>
<td align="right">0.24438461661338806</td>
</tr>
<tr>
<td>2019-11-30-12-27-19</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.35</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.8732220871107919</td>
<td align="right">0.6469451189041138</td>
<td align="right">0.1727668195962906</td>
</tr>
<tr>
<td>2019-11-30-13-08-27</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.4</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.7997605204582214</td>
<td align="right">0.6591260433197021</td>
<td align="right">0.22051429748535156</td>
</tr>
<tr>
<td>2019-11-30-13-49-20</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.45</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.9151817985943385</td>
<td align="right">0.5980278253555298</td>
<td align="right">0.21889016032218933</td>
</tr>
<tr>
<td>2019-11-30-14-30-22</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.5</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.8423596279961723</td>
<td align="right">0.639887809753418</td>
<td align="right">0.19568832218647003</td>
</tr>
<tr>
<td>2019-12-01-09-57-43</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">50</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.6937589389937264</td>
<td align="right">0.44932326674461365</td>
<td align="right">0.31281211972236633</td>
</tr>
<tr>
<td>2019-12-01-11-14-48</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">20</td>
<td align="right">64</td>
<td align="right">0.2</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.732049720627921</td>
<td align="right">0.4423047602176666</td>
<td align="right">0.3600347638130188</td>
</tr>
<tr>
<td>2019-12-01-11-54-07</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">20</td>
<td align="right">64</td>
<td align="right">0.15</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.6047670585768563</td>
<td align="right">0.42602476477622986</td>
<td align="right">0.4152842164039612</td>
</tr>
<tr>
<td>2019-12-01-12-46-13</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td> </td>
<td align="right"></td>
<td align="right"></td>
<td align="right">1.6176911507334029</td>
<td align="right">0.4225113093852997</td>
<td align="right">0.4178585410118103</td>
</tr>
<tr>
<td>2019-12-01-16-03-38</td>
<td>adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.0010000000474974513</td>
<td align="right">2.5266895975385393</td>
<td align="right">0.21345706284046173</td>
<td align="right">0.16173763573169708</td>
</tr>
<tr>
<td>2019-12-03-14-44-59</td>
<td> </td>
<td align="right"></td>
<td align="right">1</td>
<td align="right">64</td>
<td align="right">0.5</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.0010000000474974513</td>
<td align="right">4.594249248504639</td>
<td align="right">0.06111111119389534</td>
<td align="right">0.5</td>
</tr>
<tr>
<td>2019-12-03-14-49-34</td>
<td>Adam</td>
<td align="right">0.0010000000474974513</td>
<td align="right">1</td>
<td align="right">64</td>
<td align="right">0.5</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.00009999999747378752</td>
<td align="right">4.000901222229004</td>
<td align="right">1</td>
<td align="right">0</td>
</tr>
<tr>
<td>2019-12-03-21-37-40</td>
<td>Adam</td>
<td align="right">0.00009999999747378752</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.2</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.00009999999747378752</td>
<td align="right">2.4264672143118724</td>
<td align="right">0.5382830500602722</td>
<td align="right">0.12369627505540848</td>
</tr>
<tr>
<td>2019-12-04-08-18-42</td>
<td>Adam</td>
<td align="right">0.00009999999747378752</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.2</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.00009999999747378752</td>
<td align="right">2.3973828894751414</td>
<td align="right">1</td>
<td align="right">0</td>
</tr>
<tr>
<td>2019-12-04-10-39-00</td>
<td>Adam</td>
<td align="right">0.00009999999747378752</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.00009999999747378752</td>
<td align="right">2.3813047238758633</td>
<td align="right">0.26450115442276</td>
<td align="right">0.41521790623664856</td>
</tr>
<tr>
<td>2019-12-04-13-02-24</td>
<td>Adam</td>
<td align="right">0.00009999999747378752</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.15</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">5.062840325491769</td>
<td align="right">0.2650811970233917</td>
<td align="right">0.4164263606071472</td>
</tr>
<tr>
<td>2019-12-04-15-30-45</td>
<td>Adam</td>
<td align="right">0.00019999999494757503</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.15</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">3.6883838176727295</td>
<td align="right">0.26160094141960144</td>
<td align="right">0.4139128625392914</td>
</tr>
<tr>
<td>2019-12-04-20-21-55</td>
<td>Adam</td>
<td align="right">0.00009999999747378752</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.15</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">5.673606293542044</td>
<td align="right">0.2685614824295044</td>
<td align="right">0.412994384765625</td>
</tr>
<tr>
<td>2019-12-04-22-55-03</td>
<td>Adam</td>
<td align="right">0.00019999999494757503</td>
<td align="right">120</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">2.9662965876715526</td>
<td align="right">0.2726218104362488</td>
<td align="right">0.4247886836528778</td>
</tr>
<tr>
<td>2019-12-05-08-43-04</td>
<td>Adam</td>
<td align="right">0.00019999999494757503</td>
<td align="right">50</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">3.2356908661978587</td>
<td align="right">0.26740139722824097</td>
<td align="right">0.41976162791252136</td>
</tr>
<tr>
<td>2019-12-05-12-56-52</td>
<td>SGD</td>
<td align="right">0.0010000000474974513</td>
<td align="right">50</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">9.570424692971367</td>
<td align="right">0.20456303656101227</td>
<td align="right">0.4426361620426178</td>
</tr>
<tr>
<td>2019-12-05-17-59-49</td>
<td>SGD</td>
<td align="right">0.009999999776482582</td>
<td align="right">50</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0.00009999999747378752</td>
<td align="right">0.00009999999747378752</td>
<td align="right">4.068480593817575</td>
<td align="right">0.26218098402023315</td>
<td align="right">0.4107709228992462</td>
</tr>
<tr>
<td>2019-12-06-09-24-37</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>true</td>
<td align="right">0</td>
<td align="right">0.00009999999747378752</td>
<td align="right">4.647962740489414</td>
<td align="right">0.035962872207164764</td>
<td align="right">0.01998259872198105</td>
</tr>
<tr>
<td>2019-12-06-12-59-09</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.00009999999747378752</td>
<td align="right">1.6766310589654105</td>
<td align="right">0.46200695633888245</td>
<td align="right">0.5122610926628113</td>
</tr>
<tr>
<td>2019-12-06-14-08-11</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.00019999999494757503</td>
<td align="right">1.579499023301261</td>
<td align="right">0.5109049081802368</td>
<td align="right">0.4672481119632721</td>
</tr>
<tr>
<td>2019-12-06-15-11-27</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.0003000000142492354</td>
<td align="right">1.621715818132673</td>
<td align="right">0.47788095474243164</td>
<td align="right">0.4965694546699524</td>
</tr>
<tr>
<td>2019-12-06-16-14-08</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">10</td>
<td align="right">10</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.0003000000142492354</td>
<td align="right">3.2658898660114835</td>
<td align="right">0.46310901641845703</td>
<td align="right">0.4724021852016449</td>
</tr>
<tr>
<td>2019-12-06-17-26-02</td>
<td>SGD</td>
<td align="right">0.07999999821186066</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.0003000000142492354</td>
<td align="right">2.237235886710031</td>
<td align="right">0.6057618260383606</td>
<td align="right">0.31624269485473633</td>
</tr>
<tr>
<td>2019-12-06-18-43-52</td>
<td>SGD</td>
<td align="right">0.4000000059604645</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.00019999999494757503</td>
<td align="right">1.9158336009298051</td>
<td align="right">0.6982792019844055</td>
<td align="right">0.2614766061306</td>
</tr>
<tr>
<td>2019-12-07-12-33-19</td>
<td>SGD</td>
<td align="right">0.800000011920929</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.07</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.00019999999494757503</td>
<td align="right">1.952180198260716</td>
<td align="right">0.26450115442276</td>
<td align="right">0.20441938936710358</td>
</tr>
<tr>
<td>2019-12-07-14-54-42</td>
<td>Adam</td>
<td align="right">0.0015999999595806003</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.00019999999494757503</td>
<td align="right">1.7255752852984838</td>
<td align="right">0.5119683146476746</td>
<td align="right">0.44594109058380127</td>
</tr>
<tr>
<td>2019-12-07-17-03-09</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.1</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.00019999999494757503</td>
<td align="right">1.7508257627487183</td>
<td align="right">0.44650039076805115</td>
<td align="right">0.5007167458534241</td>
</tr>
<tr>
<td>2019-12-07-19-42-51</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.08</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.0003000000142492354</td>
<td align="right">1.5721245748656136</td>
<td align="right">0.5044276714324951</td>
<td align="right">0.4683695137500763</td>
</tr>
<tr>
<td>2019-12-07-21-32-04</td>
<td>Adam</td>
<td align="right">0.0007999999797903001</td>
<td align="right">30</td>
<td align="right">64</td>
<td align="right">0.05</td>
<td align="right">5</td>
<td align="right">5</td>
<td>false</td>
<td align="right">0</td>
<td align="right">0.0003000000142492354</td>
<td align="right">1.6784964203834534</td>
<td align="right">0.3552010655403137</td>
<td align="right">0.530569851398468</td>
</tr>
</tbody></table>

## Results, pros and cons of our model <!--The results of these experiments and their analysis.-->
TODO


## References
[1] Girshick, R., Donahue, J., Darrell, T. and Malik, J., 2014. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 580-587).

[2] He, K., Zhang, X., Ren, S. and Sun, J., 2015. Spatial pyramid pooling in deep convolutional networks for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 37(9), pp.1904-1916.

[3] Girshick, R., 2015. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 1440-1448).

[4] Ren, S., He, K., Girshick, R. and Sun, J., 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems (pp. 91-99).

[5] Redmon, J., Divvala, S., Girshick, R. and Farhadi, A., 2016. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

[6] Redmon, J. and Farhadi, A., 2018. Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
