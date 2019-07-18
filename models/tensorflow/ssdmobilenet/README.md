Copyright:

All models are licensed under Apache 2.0 and copyrighted by Google. See the reference page for more information.

Direct link:

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz

Reference page:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Description:

We provide a collection of detection models pre-trained on the COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset and the iNaturalist Species Detection Dataset. These models can be useful for out-of-the-box inference if you are interested in categories already in those datasets. They are also useful for initializing your models when training on novel datasets.

Notes:

This frozen graph has a built-in filter that ignores predictions below 30% confidence. Accordingly, all SSDMobileNet scores must use a 30% cutoff for comparison. Using a lower comparison generally boosts the mAP score slightly, but 30% was chosen as a reasonable tradeoff for runtime.

Training Dataset:

`COCO2014` (note: MLMark uses COCO2017 for inference)

Convereted with:

`Tensorflow 1.13.1`

Input Tensor:

`image_tensor`

Ouptut Tensors: 

`num_detections`
`detection_boxes`
`detection_scores`
`detection_classes`
