This practice page is provided as an additional resource in case example3 is not functioning properly. 

However, if you still wish to proceed with example3, you are to do so.

In this practice, we will provide more detailed instructions and code for using models deployed on Triton Server, 


Furthermore, in this practice, we will provide a more detailed example code for preprocessing data according to the config.pbtxt file before deploying a model. 

We will also provide an example code for implementing a service that uses multiple models simultaneously, which participants can run and experiment with.


## Result

![img](https://user-images.githubusercontent.com/30370933/236799782-25f9a702-7d6c-40f7-9cc1-807e4ed03d5c.jpg)


In this practice, we will provide an example code for detecting objects in an uploaded photo and drawing boxes around them. Similar to the previous practices, it is important to ensure that the preprocessing and inferencing code is aligned with the server's pbtxt file.

Participants can compare this example code with the code provided in Example2 to gain a better understanding of how to implement different computer vision tasks using Triton Server. 

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

