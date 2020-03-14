# Motivation<br>
The Motication comes from the current vqa applications for visually impaired people. We want to implement a system in which users can upload a picture of the environment or a specific item and ask a question about it. The system is able to generate an answer and read out. Thus, basically our project can be divided into two parts: Visual Question Answering (VQA) and Vocie Cloning. 
<br><br>
# Models we use<br>
We use [Pythia](https://github.com/facebookresearch/pythia) as our model to complete the VQA task. <br>
Pythia is a modular framework for Visual Question Answering research, which formed the basis for the winning entry to the VQA Challenge 2018 from Facebook AI Research (FAIR)s A-STAR team. It is built on top of PyTorch.<br><br>
For Voice Cloning task, we use [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) to read out the answer<br>
The model is an implementation of [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time.<br>

# Quickstart for VQA  
## Installation  
### 1. Install pythia environment 

1. Install Anaconda.
2. Install cudnn v7.0 and cuda.9.0. You can find a tutorial [here](https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e).
3. Create environment for pythia. Run the code below in a terminal.
```bash
conda create --name vqa python=3.6

source activate vqa
pip install demjson pyyaml

pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

pip install torchvision
pip install tensorboardX

```
### 2. Clone Pythia repository  
```
git clone https://github.com/facebookresearch/pythia ~/pythia
```
### 3. Install dependencies and setup
```
cd ~/pythia
python setup.py develop
```
## Download Data  
Datasets currently supported in Pythia require two parts of data, features and ImDB. Features correspond to pre-extracted object features from an object detector. ImDB is the image database for the datasets which contains information such as questions and answers.  
For VQA task, we need to download features from COCO dataset and VQA 2.0 ImDB. We assume that all of the data is kept inside `data` folder under `pythia` root folder. If you want to use your own dataset, the dataset should be in `data` folder. This step may take some time. 
```
cd ~/pythia;
# Create data folder
mkdir -p data && cd data;

# Download and extract the features
wget https://dl.fbaipublicfiles.com/pythia/features/coco.tar.gz
tar xf coco.tar.gz

# Get vocabularies
wget http://dl.fbaipublicfiles.com/pythia/data/vocab.tar.gz
tar xf vocab.tar.gz

# Download detectron weights required by some models
wget http://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf detectron_weights.tar.gz

# Download and extract ImDB
mkdir -p imdb && cd imdb
wget https://dl.fbaipublicfiles.com/pythia/data/imdb/vqa.tar.gz
tar xf vqa.tar.gz
```  
Here vqa2 stands for the dataset VQA2.0. If you want to use other datasets like TextVQA or VizWiz. You can change it into the corresponding key words. Here is all the datasets that Pythia currently support for VQA task:  
|Dataset|Task|Key|ImDB link|Features Link|
|:---|:---|:---|:---|:---|
|VQA2.0|vqa|vqa2|[VQA 2.0 ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/vqa.tar.gz)|[COCO](https://dl.fbaipublicfiles.com/pythia/features/coco.tar.gz)|
|VizWiz|vqa|vizwiz|[VizWiz ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/vizwiz.tar.gz)|[VizWiz](https://dl.fbaipublicfiles.com/pythia/features/vizwiz.tar.gz)|
|TextVQA|vqa|textvqa|[TextVQA 0.5 ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/textvqa_0.5.tar.gz)|[Openimages](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz)|
|VisualGenome|vqa|visual_genome|Automatically downloaded|Automatically downloaded|
|CLEVR|vqa|clevr|Automatically downloaded|Automatically downloaded|
## Training
After downloading and unzipping the data, we can start training the model
```
cd ~/pythia;
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config \
configs/vqa/vqa2/pythia.yml
```  
## Pretrain models
Performing inference using pretrained models in Pythia is easy. This section expects that you have already installed the required data as explained before.  
Here is the links to the pretrain models:  
[vqa2_train_val](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth)  
[vqa2_train_only](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia.pth)  
[vizwiz](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vizwiz/pythia_pretrained_vqa2.pth) 
We are using vqa2_train_val pretrained model. Now to run inference for EvalAI, run the following command.  
```
cd ~/pythia/data
mkdir -p models && cd models;
# Download the pretrained model
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth
cd ../..;
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config configs/vqa/vqa2/pythia_train_and_val.yml  --run_type inference --evalai_inference 1 --resume_file data/models/pythia_train_val.pth
```
If you want to train or evaluate on val, change the `run_type` to `train` or `val`
accordingly. You can also use multiple run types, for e.g. to do training, inference on
val as well as test you can set `--run_type` to `train+val+inference`.

if you remove `--evalai_inference` argument, Pythia will perform inference and provide results directly on the dataset. Do note that this is not possible in case of test sets as we don't have answers/targets for them. So, this can be useful for performing inference
on val set locally.

After the evaluation, you could found the prediction report results like this in the folder `/pythia/save/vqa_vqa2_pythia/reports`. 
```
[{"question_id": 169624000, "answer": "yes"},
 {"question_id": 93006000, "answer": "car"},
 {"question_id": 46565001, "answer": "surfboard"},
 {"question_id": 13457004, "answer": "white"},
 {"question_id": 243145000, "answer": "florida"},
 {"question_id": 402159003, "answer": "blue and white"},
 {"question_id": 155875004, "answer": "graffiti"},
 {"question_id": 24226001, "answer": "blue"},
 {"question_id": 209024002, "answer": "fast"},
 {"question_id": 365644003, "answer": "black"},
 {"question_id": 428038002, "answer": "yes"},
 {"question_id": 133130004, "answer": "batman"},
 {"question_id": 72711000, "answer": "no"},
 {"question_id": 371925002, "answer": "yes"},
 {"question_id": 364999018, "answer": "playing frisbee"},
 {"question_id": 557744002, "answer": "no"}]
```
# Demo for VQA
To quickly tryout a model interactively with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
1. Download our pythia repository.
2. Build the docker using Dockerfile in the folder `pythia`. Or you can pull our docker image from docker hub.
```bash
docker pull 
```
3. Run the docker `pythia:latest` to open a jupyter notebook with a demo model to which you can ask questions interactively.
```bash
nvidia-docker build pythia -t pythia:latest
docker run --gpus 0 -it -p 8888:8888 pythia:latest
```
The demo on jupyter notebook will kook like this:

And you will get the answers in a order based on the confidence of the prediction.

4. For your local device, you should run the commands to get the access to your Jupyter notebook.
```
ssh -i thisIsmyKey.pem -L 8888:localhost:8888 ubuntu@ec2–34–227–222–100.compute-1.amazonaws.com
```
Here is the Dockerfile.
```
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
```


# Quickstart for Voice Cloning   
You can either run the demo on localhost or on AWS.   
## Install dependencies  
You will need the following whether you plan to use the demo only or to retrain the models.
Python 3.7. Python 3.6 might work too, but I wouldn't go lower because I make extensive use of pathlib.
Run `pip install -r requirements.txt` to install the necessary packages. Additionally you will need [PyTorch](https://pytorch.org/get-started/locally/) (>=1.0.1).
A GPU is mandatory, but you don't necessarily need a high tier GPU if you only want to use the toolbox.  

## Pretrained models
Download the latest [here](https://github.com/CorentinJ/Real-Time-Voice-Cloning/wiki/Pretrained-models).  

## Preliminary
Before you download any dataset, you can begin by testing your configuration with:
`python demo_cli.py`
If all tests pass, you're good to go.

## Dataset  
You can download the dataset here [`LibriSpeech/train-clean-100`](http://www.openslr.org/resources/12/train-clean-100.tar.gz). Extract the contents as `<datasets_root>/LibriSpeech/train-clean-100` where `<datasets_root>` is a directory of your choosing. **The input of the data should be in flac/wav/m4a/mp3 format**.   

After training, you could get three models named *pretrained.pt* in the folders /encoder/, /synthesizer/, and/vocoder/.

# Demo for Voice Cloning
You can either run the demo directly using commands:
```
python demo_cli.py
```
or run it using docker.   
If you want to use docker, here are the steps:
1. You should probably have access to a machine with a CUDA-compatible GPU  
2. Install nvidia-docker  
Follow the instructions here: https://github.com/NVIDIA/nvidia-docker. Note that you’ll need have installed the NVIDIA driver and Docker as well.
3. Create Dockerfile
You can create a Dockerfile like this 
```
FROM pytorch/pytorch

WORKDIR "/workspace"
RUN apt-get clean \
        && apt-get update \
        && apt-get install -y ffmpeg libportaudio2 openssh-server python3-pyqt5 xauth \
        && apt-get -y autoremove \
        && mkdir /var/run/sshd \
        && mkdir /root/.ssh \
        && chmod 700 /root/.ssh \
        && ssh-keygen -A \
        && sed -i "s/^.*PasswordAuthentication.*$/PasswordAuthentication no/" /etc/ssh/sshd_config \
        && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
        && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
        && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config
ADD Real-Time-Voice-Cloning/requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt
CMD ["python","demo_cli.py"]
```
4. Build the docker image
run command:
```
nvidia-docker build -t pytorch-voice .
```
5. Build a container to run the demo
run command:
```
nvidia-docker run pytorch-voice
```
If you get a message says *Saved output as demo_output_00.wav*, then congratulations you run the demo successfully. Here we set the default target voice file as test.flac, and set the default text to clone as "The answer of the question is that there are three animals in the picture".  
