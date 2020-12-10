# Caffe to TensorFlow

Convert [Caffe](https://github.com/BVLC/caffe/) models to [TensorFlow 2.x](https://github.com/tensorflow/tensorflow).

## Working environment
- Ubuntu 18.04
- Tensorflow 2.3
- Caffe and Pycaffe

## Prerequisistes 
Install caffe 
For Ubuntu
```bash
apt install caffe-cpu
apt install python3-caffe-cpu
pip3 install --upgrade scikit-image
```


## Usage

Run `convert.py` to convert an existing Caffe model to TensorFlow.
```bash
python3 convert.py <prototxt file> --caffemodel <caffemodel file> --data-output-path <data file in numpy format> --code-output-path <filename to save graph as python class> --phase train --standalone-output-path <path to save savedmodel> --input_node <innode> --output_node <onode1>,<onode2>
```

Make sure you're using the latest Caffe format (see the notes section for more info).

The output consists of 3 files:

1. A data file (in NumPy's native format) containing the model's learned parameters.
2. A Python class that constructs the model's graph.
3. Tensorflow SavedModel file

## Notes

- Only the new Caffe model format is supported. If you have an old model, use the `upgrade_net_proto_text` and `upgrade_net_proto_binary` tools that ship with Caffe to upgrade them first. Also make sure you're using a fairly recent version of Caffe.

- It appears that Caffe and TensorFlow cannot be concurrently invoked (CUDA conflicts - even with `set_mode_cpu`). This makes it a two-stage process: first extract the parameters with `convert.py`, then import it into TensorFlow.

- Pycaffe is strictly required. 

- Only a subset of Caffe layers and accompanying parameters are currently supported.

- The border values are handled differently by Caffe and TensorFlow. However, these don't appear to affect things too much.

- Image rescaling can affect the ILSVRC2012 top 5 accuracy listed above slightly. VGG16 expects isotropic rescaling (anisotropic reduces accuracy to 88.45%) whereas BVLC's implementation of GoogLeNet expects anisotropic (isotropic reduces accuracy to 87.7%).

- The support class `kaffe.tensorflow.Network` has no internal dependencies. It can be safely extracted and deployed without the rest of this library.

## Credits 
1. https://github.com/ethereon/caffe-tensorflow
2. https://github.com/dhaase-de/caffe-tensorflow-python3
3. https://github.com/linkfluence/caffe-tensorflow
4. https://github.com/davidsandberg/caffe-tensorflow
