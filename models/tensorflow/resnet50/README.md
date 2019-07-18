Copyright:

All models are licensed under Apache 2.0 and copyrighted by Google. See the reference page for more information.

Reference page:

https://github.com/tensorflow/models/tree/master/research/slim

Description:

Neural nets work best when they have many parameters, making them powerful function approximators. However, this means they must be trained on very large datasets. Because training models from scratch can be a very computationally intensive process requiring days or even weeks, we provide various pre-trained models, as listed below. These CNNs have been trained on the ILSVRC-2012-CLS image classification dataset.

In the table below, we list each model, the corresponding TensorFlow model file, the link to the model checkpoint, and the top 1 and top 5 accuracy (on the imagenet test set). Note that the VGG and ResNet V1 parameters have been converted from their original caffe formats (here and here), whereas the Inception and ResNet V2 parameters have been trained internally at Google. Also be aware that these accuracies were computed by evaluating using a single image crop. Some academic papers report higher accuracy by using multiple crops at multiple scales.

Training Dataset:

`ILSVRC2012`

Input Tensor:

`input`

Ouptut Tensor: 

`resnet_v1_50/SpatialSqueeze`
