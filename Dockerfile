FROM python:3
# FROM continuumio/miniconda3
# FROM conda/miniconda3
# FROM pytorch/pytorch
# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

### python ###
RUN pip install --upgrade pip

# requirements
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt

# graphvis
RUN apt update
RUN apt upgrade --assume-yes
RUN apt install graphviz --assume-yes