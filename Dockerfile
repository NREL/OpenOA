FROM ubuntu:14.04

COPY . /openoa

RUN apt-get update
RUN yes | apt-get install python python-pip
RUN pip install -r ./openoa/requirements.txt
RUN python /openoa/setup.py install

