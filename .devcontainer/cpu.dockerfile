FROM ultralytics/ultralytics:latest-cpu
# 定义用户和用户组
ARG USERNAME=Elaina
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG GROUP_NAME=wheel
RUN apt-get update \
    && apt-get install -y  sudo vim 
# 创建用户和用户组
RUN groupadd --gid $USER_GID ${GROUP_NAME} \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    # 配置密码
    && echo "$USERNAME:password" | chpasswd \
    && usermod -aG ${GROUP_NAME} $USERNAME \
    && echo "%${GROUP_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    # 使新用户的 `.bashrc` 文件生效
    && chown $USERNAME:$GROUP_NAME /home/$USERNAME/.bashrc 


#安装处理依赖与通信库
USER root
RUN pip uninstall opencv-python opencv-python-headless ultralytics -y 
USER $USERNAME
RUN pip install  opencv-python  imutils pyzmq roslibpy ultralytics
RUN echo "export PYTHONPATH=~/yolo:$PYTHONPATH" >> ~/.bashrc