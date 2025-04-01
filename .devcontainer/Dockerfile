FROM ultralytics/ultralytics:8.3.99
# 定义用户和用户组
ARG USERNAME=Elaina
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG GROUP_NAME=wheel
RUN apt-get update \
    && apt-get install -y  sudo nautilus wget software-properties-common nautilus curl 
# 创建用户和用户组
RUN groupadd --gid $USER_GID ${GROUP_NAME} \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    # 配置密码
    && echo "$USERNAME:password" | chpasswd \
    && usermod -aG ${GROUP_NAME} $USERNAME \
    && echo "%${GROUP_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    # 使新用户的 `.bashrc` 文件生效
    && chown $USERNAME:$GROUP_NAME /home/$USERNAME/.bashrc 

#安装fish
RUN add-apt-repository ppa:fish-shell/release-3 -y \
    && apt-get update && sudo apt-get install -y fish fzf 
#安装fish插件
# RUN curl -L github.com/oh-my-fish/oh-my-fish/raw/master/bin/install | fish --noninteractive \
#     && fish -c "omf install bass" 
USER $USERNAME
RUN curl -L github.com/oh-my-fish/oh-my-fish/raw/master/bin/install | sudo tee install_omf > /dev/null \
    && fish install_omf --noninteractive \
    && fish -c "omf install bass" \
    && sudo rm install_omf 
COPY packages/fish /home/${USERNAME}/.config/fish
#安装最简ros2
USER root
ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y tzdata ros-humble-ros-base  python3-colcon-common-extensions  ros-humble-image-transport ros-humble-cv-bridge ros-humble-vision-msgs
#安装opencv和其他依赖
USER $USERNAME
RUN pip install  opencv-python  imutils
# RUN apt-get update \
#     #安装fish终端
#     && add-apt-repository ppa:fish-shell/release-3 -y \
#     && apt-get update && sudo apt-get install -y fish fzf 
# #安装fish插件
# USER $USERNAME
# RUN curl -L https://get.oh-my.fish | sudo tee install_omf > /dev/null \
#     && fish install_omf --noninteractive \
#     && fish -c "omf install bass" \
#     && sudo rm install_omf 
# USER root
