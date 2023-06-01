from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
from pytube import YouTube
import os
from werkzeug.utils import secure_filename
from inference import preprocess, infer, draw_boxes_on_image
import time
import shutil
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = os.getcwd() + '/uploads'
DOWNLOAD_FOLDER = "/Users/inyong/Desktop/model_repository"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'flv', 'wmv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the labels
with open("labels.txt", "r") as f:
    labels_map = [line.strip() for line in f]


# 초당 몇 프레임 단위로 보여줄지 정합니다. 
FRAME_RATE = 10
# 객체 정확도 임계값 입니다. 
DETECTION_THRESHOLD = 0.3 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Use a uuid as the new filename
            filename = str(uuid.uuid4()) + "." + file.filename.rsplit('.', 1)[1].lower()
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('video_feed', filename=filename))
    return render_template('index.html')


#메인화면을 랜더링하는 부분입니다.
@app.route('/')
def index():
    return render_template('index.html')

# Get the URL from the submitted form data
@app.route('/', methods=['POST'])
def submit():
    url = request.form.get('url')  

    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    # 동영상에 한글이나 특수문자가 있을 경우 제대로 인식되지 않을때까 있습니다. uuid라는 모듈을 이용하여 이름을 임의의 해쉬값으로 변경합니다.
    filename = str(uuid.uuid4()) + ".mp4" 
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    stream.download(output_path=UPLOAD_FOLDER, filename=filename)
    return render_template('stream.html', filename=filename, path='download')


#작업한 결과물을 result folder에 저장해주기 위해 result 폴더를 생성해줍니다.
RESULT_FOLDER = os.getcwd() + '/result'

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace(' ', '_'))  

    camera = cv2.VideoCapture(video_path)  # Open the video file
    total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)  # Adjust the frame rate


    # 작업한 프레임들을 다시 영상으로 변환해주기 위한 코드
    video_FourCC = int(camera.get(cv2.CAP_PROP_FOURCC))
    video_fps = int(camera.get(cv2.CAP_PROP_FPS))
    video_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    result_path = os.path.join(RESULT_FOLDER, filename.replace(' ', '_'))  
    videoWriter = cv2.VideoWriter(result_path, video_FourCC, video_fps, video_size)

    # Create a list to hold the processed frames
    processed_frames = []

    current_frame = 0  # Initialize current frame number
    start_time = time.time()  # Record the start time
    frame_start_time = start_time

    while True:
        success, frame = camera.read()  # read the video frame
        if not success:
            camera.release()  # If the video is over, release the video file
            break
        else:
            # Save the current frame to a temporary file
            cv2.imwrite("temp.jpg", frame)
            # Preprocess the image
            input_batch = preprocess("temp.jpg")
            # inference.py의 infer 함수에 이미지를 요청합니다
            num_detections, detection_classes, detection_scores, detection_boxes = infer(input_batch)
            # Draw the boxes on the image
            output_frame = draw_boxes_on_image(frame, detection_boxes[0], detection_classes[0], detection_scores[0], labels_map)
            end_time = time.time()
            frame_processing_time = end_time - frame_start_time
            frame_start_time = end_time  
            # 작업 끝날때까지 남은 시간
            print(f"Processed frame {len(processed_frames)}/{total_frames} ({100 * len(processed_frames) / total_frames:.2f}%) in {frame_processing_time:.2f} seconds.")


            #탐지된 객체에 따라 출력을 보여줍니다
            if num_detections > 0:
                detected_classes = [labels_map[int(detection_classes[0][i])] for i in range(len(detection_classes[0])) if detection_scores[0][i] >= DETECTION_THRESHOLD]
                if len(detected_classes) > 0:
                    print(f"Detected objects: {detected_classes}")
                else:
                    print(f"No objects detected with accuracy over {DETECTION_THRESHOLD}")
            else:
                print("No objects detected.")

            videoWriter.write(output_frame)
            processed_frames.append(output_frame)
            # frame number 증가
            current_frame += 1  

        if int(end_time - start_time) >= 10:  # 10 seconds have passed
            fps = current_frame / (end_time - start_time)
            print(f"Average FPS in the last 10 seconds: {fps:.2f}")
            start_time = end_time  # Reset the start time
            current_frame = 0  # Reset the frame count

    # 작업 완료시 동영상으로 저장해서 다시 볼 수 있도록 합니다.
    videoWriter.release()
    
    os.remove("temp.jpg")

    # 동영상 처리가 끝나면 실제 index.html에 변환된 동영상을 출력해준다.
    def generate_frames():
        for frame in processed_frames:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
