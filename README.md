# Problem<br>
The problem that we are focusing is an application for visually impaired people. We want to complete a system in which they can take a photo of the environment or a specific item and ask a question about it like "What is this?". Idealy the system will say the answer clearly. So basically our project can be divided into two parts: Visual Question Answering (VQA) and Vocie Cloning. 
<br><br>
# How to solve<br>
We use Pythia as our model to complete the VQA task. <br>
https://github.com/facebookresearch/pythia<br>
Pythia is a modular framework for Visual Question Answering research, which formed the basis for the winning entry to the VQA Challenge 2018 from Facebook AI Research (FAIR)s A-STAR team. It is built on top of PyTorch.<br><br>
For Voice Cloning part, we use the method below:<br>
https://github.com/CorentinJ/Real-Time-Voice-Cloning<br>
The method is based on  [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) with a vocoder that works in real-time.<br>

# Quickstart for VQA  
## Installation  
### 1. Clone Pythia repository  
```
git clone https://github.com/facebookresearch/pythia ~/pythia
```
### 2. Install dependencies and setup
```
cd ~/pythia
python setup.py develop
```
## Download Data  
Datasets currently supported in Pythia require two parts of data, features and ImDB. Features correspond to pre-extracted object features from an object detector. ImDB is the image database for the datasets.  
Note: Change the feature link and ImDB link according to the dataset you are using. 
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
If you want to use your own dataset, the data in the dataset should be in 
## Training
After downloading and unzipping the data, we can start training the model
```
cd ~/pythia;
python tools/run.py --tasks vqa --datasets vqa2 --model pythia --config \
configs/vqa/vqa2/pythia.yml
```  
Here vqa2 stands for the dataset VQA2.0. If you want to use other datasets like TextVQA or VizWiz. You can change it into the corresponding key words. Here is all the datasets that Pythia currently support for VQA task:  
|Dataset|Task|Key|ImDB link|Features Link|
|:---|:---|:---|:---|:---|
|VQA2.0|vqa|vqa2|[VQA 2.0 ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/vqa.tar.gz)|[COCO](https://dl.fbaipublicfiles.com/pythia/features/coco.tar.gz)|
|VizWiz|vqa|vizwiz|[VizWiz ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/vizwiz.tar.gz)|[VizWiz](https://dl.fbaipublicfiles.com/pythia/features/vizwiz.tar.gz)|
|TextVQA|vqa|textvqa|[TextVQA 0.5 ImDB](https://dl.fbaipublicfiles.com/pythia/data/imdb/textvqa_0.5.tar.gz)|[Openimages](https://dl.fbaipublicfiles.com/pythia/features/open_images.tar.gz)|
|VisualGenome|vqa|visual_genome|Automatically downloaded|Automatically downloaded|
|CLEVR|vqa|clevr|Automatically downloaded|Automatically downloaded|
## Pretrain models
Performing inference using pretrained models in Pythia is easy. This section expects that you have already installed the required data as explained before.  
Here is the links to the pretrain models:  
[vqa2_train_val](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia_train_val.pth)  
[vqa2_train_only](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vqa2/pythia.pth)  
[vizwiz](https://dl.fbaipublicfiles.com/pythia/pretrained_models/vizwiz/pythia_pretrained_vqa2.pth)  
Now, let's say your link to pretrained model `model` is `link` (select from table > right click > copy link address), the respective config should be at
`configs/[task]/[dataset]/[model].yml`. For example, config file for `vqa2 train_and_val` should be
`configs/vqa/vqa2/pythia_train_and_val.yml`. Now to run inference for EvalAI, run the following command.  
Here is an example for vqa2_train_val:
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

if you remove `--evalai_inference` argument, Pythia will perform inference and provide results
directly on the dataset. Do note that this is not possible in case of test sets as we
don't have answers/targets for them. So, this can be useful for performing inference
on val set locally.

After the evaluation, you could found the prediction like this in the folder /save/.  


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
