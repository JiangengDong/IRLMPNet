FROM nvidia/cuda:10.2-runtime-ubuntu18.04

# dependencies
RUN apt-get update && \
    apt-get install -y apt-utils sudo dialog wget lsb-release vim build-essential git unzip cmake locales \
                       python3-pip python3-distutils python3-dev libboost-all-dev libeigen3-dev llvm-10-dev && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
ENV LLVM_CONFIG=/usr/bin/llvm-config-10 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en

# add a user "arclab" with password "arclab"
RUN useradd -m arclab && \
    echo "arclab:arclab" | chpasswd && \
    adduser arclab sudo
# change user and workdir
USER arclab
WORKDIR /home/arclab
RUN pip3 install wheel pylint rope autopep8 numpy
