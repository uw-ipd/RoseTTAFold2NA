FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#COPY . /home/ProteinMPNN
#ADD dockerrequirements.txt /requirements.txt

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt-get install python3 -y\
    && apt-get install python3-pip -y\
    && apt-get install -y git\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install wandb==0.12.0 pynvml==11.0.0 
RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
RUN pip install scipy requests packaging e3nn
RUN pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html

ADD SE3Transformer /SE3Transformer
RUN cd SE3Transformer/ && python3 setup.py install
RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN pip install -r /requirements.txt
#ADD requirements.txt /home/requirements.txt
#RUN pip install -r /home/requirements.txt
#WORKDIR /workdir/
#RUN wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz
#RUN tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz
RUN git clone https://github.com/soedinglab/hh-suite.git\
    && mkdir -p hh-suite/build && cd hh-suite/build\
    && cmake -DCMAKE_INSTALL_PREFIX=. ..\
    && make -j 4 && make install

ENV PATH="${PATH}:/hh-suite/build/bin/"

RUN apt-get update \
    && apt-get install -y vim
