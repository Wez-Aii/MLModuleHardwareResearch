FROM nvcr.io/nvidia/cuda:11.4.3-runtime-ubuntu20.04
RUN apt-get update -y
RUN apt-get install -y python3-pip
RUN pip install fastapi==0.88.0
RUN pip install albumentations==1.3.0
RUN pip install torchinfo==1.8.0
RUN pip install lightning==2.0.2
RUN pip install clearml==1.11.0
RUN pip install segmentation-models-pytorch==0.3.3
RUN pip install transformers==4.29.2
RUN pip install Pillow==9.2.0
RUN pip install funcy==2.0
RUN pip install trains==0.16.4
RUN pip install tenacity==8.2.2
RUN pip install tensorboardX==2.6
RUN pip install protobuf==3.20.3
RUN pip install argparse
RUN pip install scikit-multilearn
RUN pip install matplotlib
RUN pip install onnx
RUN pip install onnxruntime
RUN apt-get update -y
RUN apt-get install -y git
RUN apt-get install -y unzip
RUN apt-get install -y zip
RUN apt-get -y install curl
RUN apt-get -y install nano
RUN pip install pycocotools
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt