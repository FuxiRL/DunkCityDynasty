FROM nvidia/cuda:11.4.0-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y vim
RUN apt-get install -y wget
RUN apt-get install -y x11vnc
RUN apt-get install -y xvfb
RUN apt-get install -y net-tools

# install wine
RUN dpkg --add-architecture i386
RUN apt-get update
RUN wget -qO- https://dl.winehq.org/wine-builds/winehq.key | apt-key add -
RUN apt install -y software-properties-common
RUN apt-add-repository "deb http://dl.winehq.org/wine-builds/ubuntu/ $(lsb_release -cs) main"
RUN apt install -y --install-recommends winehq-stable

# install pip dependencies
RUN echo "Installing pip dependencies..."
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3.8-dev python3.8-distutils python3.8-venv
RUN wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py
RUN python3.8 /tmp/get-pip.py
RUN pip3.8 install numpy==1.22.4
RUN pip3.8 install tensorflow==2.9.1
RUN pip3.8 install torch==1.11.0
RUN pip3.8 install ray==2.3.0
RUN pip3.8 install ray[rllib]==2.3.0
RUN pip3.8 install docker==5.0.3
RUN pip3.8 install pypng==0.0.21
RUN pip3.8 install imageio==2.19.3
RUN pip3.8 install protobuf==4.22.3
RUN pip3.8 install gymnasium==0.26.3
RUN pip3.8 install opencv-python==4.5.5.64
RUN pip3.8 install grpcio==1.54.0 
RUN pip3.8 install grpcio-tools==1.54.0 
RUN pip3.8 install jsonlines==3.1.0

