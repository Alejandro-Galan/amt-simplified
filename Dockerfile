FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub


RUN apt update --fix-missing
RUN apt install -y python3.7 python3.7-distutils

RUN apt install build-essential -y
RUN apt install ffmpeg libsm6 vim -y
RUN DEBIAN_FRONTEND=noninteractive apt install python3-opencv -y
RUN apt install default-jdk fluidsynth -y
RUN apt clean

RUN pip install --upgrade pip

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# CUDA 10.1 and cuDNN 7.6
RUN apt-get update && apt-get install -y \
    cuda-toolkit-10-1 \
    libcudnn7=7.6.5.32-1+cuda10.1 \
    && apt-get clean
