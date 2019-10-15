The three models in this folder were created by converting the "golden" MLMark models in the `tensorflow` folder using the scripts in the `tensorflow_lite/utility` folder. They were converted to TensorFlow LITE and then quantized to `int8` using PTIQ on 200 images from the relevant data set.

The "edgetpu" models here were compiled with the [Google Edge TPU Compiler](https://coral.withgoogle.com/docs/edgetpu/compiler/).

Please refere to the `tensorflow` models folder for copyright information.
