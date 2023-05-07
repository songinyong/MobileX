
# install environment
sudo pip3 install flask tritionclient geventhttpclient


# Execution command

Change directory to the example2 folder and execute the command "python3 app.py"


## 실행 화면
<img width="707" alt="스크린샷 2023-05-07 오후 7 10 17" src="https://user-images.githubusercontent.com/30370933/236671302-7224fdc8-6647-4e6e-ae58-6b036e14d7ce.png">



This tutorial demonstrates how to create a simple service that identifies objects in an image using Triton Inference Server.

The advantage of Triton server is that it provides a unified API for models using various frameworks such as PyTorch or TensorFlow.

outputs = httpclient.InferRequestedOutput("fc6_1", binary_data=True)

start_time = time.time()
res = client.infer(model_name="densenet_onnx", inputs=[inputs], outputs=[outputs])
result = res.as_numpy("fc6_1")
end_time = time.time()

해당 
