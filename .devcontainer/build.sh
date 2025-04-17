#!/bin/bash

# 检查是否传入参数
if [ "$1" == "--cpu" ]; then
    # 如果是 --cpu 参数，使用 cpu.dockerfile
    docker build -t elaina/yolo_image -f cpu.dockerfile .
#如果传入ros
elif [ "$1" == "--ros" ]; then
    # 如果是 --ros 参数，使用 ros.dockerfile
    docker build -t elaina/yolo_image -f ros.dockerfile .
else
    # 默认情况，使用 Dockerfile（假设该文件是默认的）
    docker build -t elaina/yolo_image .
fi
