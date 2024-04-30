FROM mambaorg/micromamba:1.5.0 as micromamba
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Create root owned env: https://github.com/mamba-org/micromamba-docker/blob/main/examples/add_micromamba/Dockerfile
USER root
ENV MAMBA_USER=root
ENV MAMBA_USER_ID=0
ENV MAMBA_USER_GID=0
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]



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

ADD . /RF2NA/
#ADD SE3Transformer /SE3Transformer
RUN cd /RF2NA/SE3Transformer/ && python3 setup.py install
RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN pip install -r /requirements.txt
#ADD requirements.txt /home/requirements.txt
#RUN pip install -r /home/requirements.txt
#WORKDIR /workdir/
#RUN wget https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-AVX2-Linux.tar.gz
#RUN tar xvfz hhsuite-3.3.0-AVX2-Linux.tar.gz
RUN cd /RF2NA/network/\
    && wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz\
    && tar xvfz RF2NA_apr23.tgz
RUN git clone https://github.com/soedinglab/hh-suite.git\
    && mkdir -p hh-suite/build && cd hh-suite/build\
    && cmake -DCMAKE_INSTALL_PREFIX=. ..\
    && make -j 4 && make install

ENV PATH="${PATH}:/hh-suite/build/bin/"

RUN apt-get update \
    && apt-get install -y vim

RUN pip install pandas
RUN pip install pydantic
ENV HH_DB=/mnt/databases/pdb100_2021Mar03/pdb100_2021Mar03
ENV DB_UR30=/mnt/databases/UniRef30_2020_06/UniRef30_2020_06
ENV DB_BFD=/mnt/databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt
