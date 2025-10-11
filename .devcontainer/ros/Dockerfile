#FROM elainasuki/ros:ros2-humble-full-v3
FROM elainasuki/ros:ros2-humble-full-0614

ARG USERNAME=Elaina

ARG USER_GID=$USER_UID

ARG GROUP_NAME=wheel
RUN apt-get update \
    && apt-get install -y  sudo vim nautilus 

USER ${USERNAME}

# 安装fish插件（切换到普通用户）
RUN curl -L https://github.com/oh-my-fish/oh-my-fish/raw/master/bin/install | sudo tee install_omf > /dev/null && \
    fish install_omf --noninteractive && \
    fish -c "omf install bass" && \
    sudo rm install_omf 

# 复制fish配置
COPY packages/fish /home/${USERNAME}/.config/fish

#安装前置依赖
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装yolo
RUN pip install imutils ultralytics openvino
#安装其他依赖
RUN pip install zmq "numpy<2.0"
# 安装ROS通信库
RUN pip install roslibpy pyzmq

USER root
RUN echo "export PYTHONPATH=~/yolo:$PYTHONPATH" >> ~/.bashrc \
    && echo "export ROS_LOCALHOST_ONLY=1" >> /home/$USERNAME/.bashrc \
    && echo "export PYTHONPATH=/home/Elaina/yolo:$PYTHONPATH" >> /home/$USERNAME/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc \
    && echo "export PATH=\$HOME/.local/bin:\$PATH" >> /home/$USERNAME/.bashrc \
    && echo "set -gx PATH \$HOME/.local/bin \$PATH" >> /home/$USERNAME/.config/fish/config.fish
