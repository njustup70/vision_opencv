FROM elainasuki/ros:ros2-humble-full-v3
ARG USERNAME=Elaina
USER ${USERNAME}
#安装前置依赖
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# 安装yolo
RUN pip install imutils ultralytics openvino
#安装其他依赖
RUN pip install zmq "numpy<2.0"