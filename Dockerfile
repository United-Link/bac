FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Add default values for USER_GID, USER_UID, and USER_NAME
ARG USER_NAME=defaultuser
ARG USER_UID=1000
ARG USER_GID=1000

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

RUN apt update
RUN apt install -y \
    sudo \
    python-is-python3 \
    python3-pip \
    ffmpeg \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-x \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgl1 \
    libgl1-mesa-glx \
    dkms \
    cmake \
    gcc \
    g++ \
    git \
    vim \
    wget \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libopenblas-base \
    libopenblas-dev \
    libatlas-base-dev

RUN apt autoremove

# Add user and group, providing default values
RUN groupadd -g $USER_GID $USER_NAME && \
    useradd -m -u $USER_UID -g $USER_NAME -s /bin/bash $USER_NAME

# Add user to sudoers with no password prompt
RUN usermod -aG sudo $USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN usermod -aG video $USER_NAME

USER $USER_NAME
WORKDIR /home/$USER_NAME

RUN pip install numpy==1.26.4

# build opencv
RUN git clone -b 4.9.0 https://github.com/opencv/opencv.git
RUN git clone -b 4.9.0 https://github.com/opencv/opencv_contrib

RUN mkdir /home/$USER_NAME/opencv/build
WORKDIR /home/$USER_NAME/opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/home/$USER_NAME/.local \
          -D INSTALL_C_EXAMPLES=OFF \
          -D INSTALL_PYTHON_EXAMPLES=OFF \
          -D PYTHON_EXECUTABLE=$(which python3) \
          -D BUILD_opencv_python2=OFF \
          -D PYTHON3_EXECUTABLE=$(which python3) \
          -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
          -D PYTHON3_PACKAGES_PATH=/home/$USER_NAME/.local/lib/python3.10/site-packages \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
          -D WITH_GSTREAMER=ON \
          -D BUILD_EXAMPLES=OFF ..

RUN make -j$(nproc) && make install

WORKDIR /home/$USER_NAME

COPY --chown=$USER_NAME:$USER_NAME requirements.txt .
RUN pip install -r requirements.txt
RUN rm -rf ./requirements.txt opencv opencv_contrib

