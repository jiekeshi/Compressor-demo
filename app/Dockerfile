# app/Dockerfile

FROM ubuntu:18.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt -y upgrade
RUN apt-get -y update
RUN apt -y install build-essential software-properties-common curl git vim htop tmux wget

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt -y upgrade
RUN apt-get -y update
RUN apt -y install python3.9 python3-pip python3.9-distutils python3.9-dev

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
# Be careful with the torch version that you need to use, decide it according to your cuda version
RUN pip3 install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install transformers==4.21.1 scikit-learn==1.1.2 tree-sitter==0.20.0 gdown==4.5.1 streamlit

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]