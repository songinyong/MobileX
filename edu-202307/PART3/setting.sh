#!/bin/bash

npm install

container_id=$(docker ps -q --filter ancestor=localhost/cam-deploy:1.0)

# 컨테이너가 실행중이면 종료
if [[ ! -z "$container_id" ]]
then
    # 컨테이너를 종료합니다.
    echo "Stopping container $container_id"
    docker stop $container_id
else
    echo "No running container found for image localhost/cam-deploy:1.0"
fi


sudo dnf -y install ffmpeg

cp webcam_streaming.txt /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/webcam_streaming.txt

# deepstream app start
sudo deepstream-app -c /opt/nvidia/deepstream/deepstream/samples/configs/tao_pretrained_models/webcam_streaming.txt
