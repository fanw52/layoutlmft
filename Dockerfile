#FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
#ENV LANG C.UTF-8
#RUN sed -i 's#http://archive.ubuntu.com#http://mirrors.163.com#g' /etc/apt/sources.list && \
#    sed -i 's#http://security.ubuntu.com#http://mirrors.163.com#g' /etc/apt/sources.list
#
#RUN apt update && \
#    apt install -y bash \
#                   build-essential \
#                   git \
#                   curl \
#                   ca-certificates \
#                   python3 \
#                   python3-pip && \
#    rm -rf /var/lib/apt/lists

#ENV PATH="/root/miniconda3/bin:${PATH}"
#ARG PATH="/root/miniconda3/bin:${PATH}"
#RUN apt-get update
#
#RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
#
#RUN wget \
#    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
#    && mkdir /root/.conda \
#    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
#    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh
#RUN conda --version

#RUN python3 -m pip install --no-cache-dir --upgrade pip
#
#WORKDIR /layoutxlm
#COPY . /layoutxlm
#
#RUN pip install Cython numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
#RUN cd ./cocoapi/PythonAPI
#RUN python3 setup.py install
#
#
#RUN pip install --no-cache-dir -r requirements.txt
#CMD ["/bin/bash"]

FROM layoutxlm:v0.0.8
WORKDIR /layoutxlm
COPY . /layoutxlm

CMD ["/bin/bash"]