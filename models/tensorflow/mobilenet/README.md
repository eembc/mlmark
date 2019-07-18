Copyright:

All models are licensed under Apache 2.0 and copyrighted by Google. See the reference page for more information.

Direct link:

http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz

Reference page:

https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md

Description:

Choose the right MobileNet model to fit your latency and size budget. The size of the network in memory and on disk is proportional to the number of parameters. The latency and power usage of the network scales with the number of Multiply-Accumulates (MACs) which measures the number of fused Multiplication and Addition operations. These MobileNet models have been trained on the ILSVRC-2012-CLS image classification dataset. Accuracies were computed by evaluating using a single image crop.

Training Dataset:

`ILSVRC2012`

Convereted with:

`Tensorflow 1.13.1`

Input Tensor:

`input`

Ouptut Tensor: 

`MobilenetV1/Predictions/Reshape_1`

Note: The output tensor shape has 1001 categories, as this model was trained with a "background" category.
