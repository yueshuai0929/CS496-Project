FROM nvidia/cuda:10.2-base
FROM python:3-stretch
FROM jupyter/datascience-notebook

# This is needed to ensure cuda can view GPU
ENV NVIDIA_DRIVER_CAPABILITIES compute, utility

RUN pip install --upgrade pip

# Download files for model
#WORKDIR "/workspace"
# RUN mkdir Pythia 
#ADD ./ /workspace
COPY pythia_demo.ipynb ./


#RUN mkdir content
#RUN cd content
RUN mkdir model_data
RUN wget -O model_data/answers_vqa.txt https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
RUN wget -O model_data/vocabulary_100k.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_100k.txt
RUN wget -O model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth
RUN wget -O model_data/pythia.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth
RUN wget -O model_data/pythia.yaml https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.yml
RUN wget -O model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
RUN wget -O model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
RUN tar xf model_data/detectron_weights.tar.gz


# Current pillow 7.0 has a compatability error
RUN pip install Pillow==6.1

# Install dependencies
RUN pip install ninja yacs cython matplotlib demjson
RUN pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

# Install fastText
RUN git clone https://github.com/facebookresearch/fastText.git fastText && cd fastText && pip install -e .

# Installing Pythia
RUN git clone https://github.com/facebookresearch/pythia.git pythia && cd pythia && pip install -e .

# Installing maskrcnn
RUN git clone https://gitlab.com/meetshah1995/vqa-maskrcnn-benchmark.git && cd vqa-maskrcnn-benchmark && python setup.py build && python setup.py develop


USER root
RUN apt-get update
RUN apt-get install software-properties-common --assume-yes
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt install nvidia-384 nvidia-modprobe --assume-yes
RUN wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
RUN chmod +x cuda_9.0.176_384.81_linux-run 
RUN ./cuda_9.0.176_384.81_linux-run --extract=$HOME
RUN ./cuda-linux.9.0.176-22781540.run -noprompt
RUN wget https://s3.amazonaws.com/open-source-william-falcon/cudnn-9.0-linux-x64-v7.1.tgz
RUN tar -xzvf cudnn-9.0-linux-x64-v7.1.tgz  
RUN cp cuda/include/cudnn.h /usr/local/cuda/include
RUN cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
RUN chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
RUN echo 'export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64\"\nexport CUDA_HOME=/usr/local/cuda\nexport PATH=$PATH:/usr/local/cuda/bin'>>~/.bashrc
#Install cuda

# Create jupyter notebook entrypoint
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]