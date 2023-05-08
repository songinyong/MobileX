import io
import requests
import time
from PIL import Image
from torchvision import transforms
import json
import tritonclient.http as httpclient

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def preprocess(image_path):
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(img)
    # Transpose the tensor to NHWC format
    input_tensor = input_tensor.permute(1, 2, 0)
    # Convert the tensor to UINT8 format
    input_tensor = (input_tensor * 255).byte()
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def infer(input_batch):
    try:
        client = httpclient.InferenceServerClient(url="triton.default.svc.ops.openark:8000")
        inputs = httpclient.InferInput("image_tensor", list(input_batch.shape), "UINT8")
        inputs.set_data_from_numpy(input_batch.numpy())
        outputs = [
            httpclient.InferRequestedOutput("num_detections", binary_data=True),
            httpclient.InferRequestedOutput("detection_classes", binary_data=True),
            httpclient.InferRequestedOutput("detection_scores", binary_data=True),
            httpclient.InferRequestedOutput("detection_boxes", binary_data=True),
        ]

        start_time = time.time()
        res = client.infer(model_name="ssd_mobilenet_v1_coco_2018_01_28", inputs=[inputs], outputs=outputs)
        num_detections = res.as_numpy("num_detections")
        detection_classes = res.as_numpy("detection_classes")
        detection_scores = res.as_numpy("detection_scores")
        detection_boxes = res.as_numpy("detection_boxes")
        end_time = time.time()

        inf_time = end_time - start_time

        print(f"Inference time: {inf_time * 1000:.3f} ms")
        print(f"Input shape: {input_batch.shape}")
        #print(f"Num detections: {num_detections}")
        #print(f"Detection classes: {detection_classes}")
        #print(f"Detection scores: {detection_scores}")
        #print(f"Detection boxes: {detection_boxes}")
        
        return num_detections, detection_classes, detection_scores, detection_boxes

    except httpclient.InferenceServerException as e:
        print("Error occurred during inference:")
        print(str(e))
        if hasattr(e, 'status_code'):
            print(e.status_code)
        if hasattr(e, 'message'):
            print(e.message)

            
def index_to_label(index):
    return labels[int(index) - 1]



def visualize_results(image_path, detection_boxes, detection_scores, detection_classes, labels_map, threshold=0.5):
    image = Image.open(image_path)
    plt.figure(figsize=(12, 12))
    plt.imshow(image)

    ax = plt.gca()

    for i in range(len(detection_scores)):
        if detection_scores[i] >= 0.2:
            box = detection_boxes[i]
            y_min, x_min, y_max, x_max = box[0] * image.height, box[1] * image.width, box[2] * image.height, box[3] * image.width

            label = index_to_label(detection_classes[i])
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)
            plt.text(x_min, y_min, f"{label}: {detection_scores[i]:.2f}", fontsize=12, color="r", bbox=dict(facecolor="r", alpha=0.5))

    plt.axis("off")
    plt.show()
    
def draw_boxes_on_image(image, detection_boxes, detection_classes, detection_scores, category_index, min_score_thresh):
    for i in range(len(detection_boxes)):
        if detection_scores[i] > min_score_thresh:
            class_id = int(detection_classes[i])
            class_name = category_index[class_id]["name"]
            score = detection_scores[i]
            box = detection_boxes[i]

            y1, x1, y2, x2 = (box * np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{class_name}: {score:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image
