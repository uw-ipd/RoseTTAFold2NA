#!/bin/bash

conda create -y --name RF2NA python==3.10.14

conda activate RF2NA

conda install -y conda-forge::llvm-openmp

conda install -y -c bioconda blast csblast cd-hit infernal mafft hmmer hhsuite

pip install torch==2.2.1 torchvision torchaudio torchdata torch_geometric

pip install pandas pydantic e3nn wandb pynvml git+https://github.com/NVIDIA/dllogger#egg=dllogger

pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

cd SE3Transformer

python setup.py install

cd ..

