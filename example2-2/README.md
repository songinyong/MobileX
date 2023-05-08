This practice page is provided as an additional resource in case example3 is not functioning properly. 

However, if you still wish to proceed with example3, you are to do so.

In this practice, we will provide more detailed instructions and code for using models deployed on Triton Server, 


Furthermore, in this practice, we will provide a more detailed example code for preprocessing data according to the config.pbtxt file before deploying a model. 

We will also provide an example code for implementing a service that uses multiple models simultaneously, which participants can run and experiment with.


## Result

<img width="718" alt="스크린샷 2023-05-08 오후 8 24 29" src="https://user-images.githubusercontent.com/30370933/236811742-417e064d-8dde-4ce6-9186-fdf86acaa04f.png">



In this practice, we will provide an example code for detecting objects in an uploaded photo and drawing boxes around them. Similar to the previous practices, it is important to ensure that the preprocessing and inferencing code is aligned with the server's pbtxt file.

Participants can compare this example code with the code provided in Example2 to gain a better understanding of how to implement different computer vision tasks using Triton Server. 


max_batch_size: Specifies the maximum batch size that can be processed at once. In this example, up to 128 pieces of data can be processed at once.

input: Specifies information about the input data. Since there can be multiple inputs, it is written in list format.

name: Specifies the name of the input data. In this example, it is set to "image_tensor".

format: Specifies the format of the input data. In this example, the NHWC format is used.

data_type: Specifies the data type of the input data. In this example, an 8-bit integer (TYPE_UINT8) is used.

dims: Specifies the dimensions of the input data. -1 means a variable length dimension. In this example, the height and width are variable, and the number of channels is 3.



output: Specifies information about the output data. Since there can be multiple outputs, it is written in list format.

name: Specifies the name of the output data. In this example, names such as "num_detections" and "detection_classes" are used.

data_type: Specifies the data type of the output data. In this example, a floating-point (TYPE_FP32) is used.

dims: Specifies the dimensions of the output data. In this example, the dimensions are set to 1 for "num_detections" and 100 for "detection_classes".

reshape: Used when the dimensions of the output data need to be reshaped. In this example, it is used for "num_detections" and is set to reshape to the default shape.

```
max_batch_size: 128
input [
  {
    name: "image_tensor"
    format: FORMAT_NHWC
    data_type: TYPE_UINT8
    dims: [ -1, -1, 3]
  }
]
output [
  {
    name: "num_detections"
    data_type: TYPE_FP32
    dims: 1
    reshape {}
  },
  {
    name: "detection_classes"
    data_type: TYPE_FP32
    dims: 100
  },
...
]
```


Next is a problem using the resnet101 model.

A pbtxt and example code are provided, so try to complete the actual code to work during the practice time. Whether to implement the code to work on an actual web page is optional.


The final result will be similar to Example2.

```
name: "resnet101"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  {
    name: "data"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "resnetv25_dense0_fwd"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

```

<img width="560" alt="스크린샷 2023-05-08 오후 8 35 59" src="https://user-images.githubusercontent.com/30370933/236813937-8e96f31c-6918-451a-9c29-b89f4d94cff3.png">

