FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04
# FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV CONDA_DIR $HOME/miniconda3
# make non-activate conda commands available
ENV PATH=$CONDA_DIR/bin:$PATH
# make conda activate command available from /bin/bash --login shells

RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

RUN apt update -y && \
    apt install wget gcc unzip curl git -y  && \
    apt -y install ffmpeg libsm6 libxext6 libglib2.0-0 libgl1-mesa-glx && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod 775 Miniconda3-latest-Linux-x86_64.sh  && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR  && \
    rm Miniconda3-latest-Linux-x86_64.sh  && \
    export PATH="$HOME/miniconda3/bin:$PATH"  && \
    conda create -n satellogic python=3.6  -q -y
    
RUN conda run -n satellogic pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

RUN conda run -n satellogic  pip install git+https://gitlab+deploy-token-28:xkxRsx2anp-u3_V4aAK9@publicgitlab.satellogic.com/iqf/iq_tool_box-@regressor-rebase-b
   
RUN conda run -n satellogic pip install piq tqdm tensorboardX && \
    conda run -n satellogic pip install imageio scikit-image

WORKDIR /regressor

CMD ["/bin/bash", "-c", "source activate satellogic && /bin/bash"]