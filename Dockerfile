FROM  nvidia/cuda:11.1-devel-ubuntu20.04


# modify source to use sjtu mirror
RUN rm /etc/apt/sources.list.d/* && \
    sed -i 's/archive.ubuntu.com/mirrors.sjtug.sjtu.edu.cn/g' /etc/apt/sources.list && \
    sed -i '/security.ubuntu.com/d' /etc/apt/sources.list

RUN apt-get update && apt-get install -y apt-utils

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 wget curl zsh fish\
        python3-distutils libglib2.0-dev python3-dev python3-pip git iputils-ping net-tools\
        ninja-build protobuf-compiler libprotobuf-dev wget ca-certificates postgresql-12\
        unzip build-essential vim openssh-server sshfs sudo\
        apt-transport-https tmux npm nodejs yarn\
        ca-certificates htop bmon iftop iotop\
        gnupg-agent autojump\
        software-properties-common \
        && ln /usr/bin/python3 /usr/bin/python

RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'     
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install apt-utils cmake swig libopenblas-pthread-dev libopenblas64-pthread-dev libpthread-stubs0-dev libblis-pthread-dev libomp-dev
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py 
RUN python get-pip.py && rm get-pip.py

RUN pip config  set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config  set global.trusted-host pypi.tuna.tsinghua.edu.cn

RUN pip install numpy jupyterlab flask peewee pandas redis xlrd \
                opencv-python-headless tqdm Pillow matplotlib pycocotools \
                sqlitedict tensorboard cython six terminaltables \
                yapf psycopg2-binary addict albumentations


RUN wget http://192.168.3.60:25566/s/isiD6GSFgRZdnq3/download -O torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl &&\
                pip install torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl &&\
                rm torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl && \
                pip install torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
