# foundation:v0

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    unzip \
    rsync \
    openssh-server \
    libhdf5-dev \
    libsm6 \
    libgl1-mesa-glx \
    graphviz \
    fish && \
    rm -rf /var/lib/apt/lists/*

RUN curl -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh \
    \
    && /opt/conda/bin/conda update -y -n base -c defaults conda \
    \
    && /opt/conda/bin/conda install -y -c conda-forge \
    python=$PYTHON_VERSION \
    av=10.0.0 \
    jupyterlab=3.6.3 \
    pandas=2.0.0 \
    h5py=3.8.0 \
    scipy=1.10.1 \
    scikit-learn=1.2.2 \
    scikit-image=0.20.0 \
    pynndescent=0.5.8 \
    numdifftools=0.9.41 \
    lmfit=1.2.0 \
    statsmodels=0.13.5 \
    networkx=3.1 \
    gitpython=3.1.31 \
    matplotlib=3.7.1 \
    seaborn=0.12.2 \
    plotly=5.14.1 \
    pillow=9.5.0 \
    umap-learn=0.5.3 \
    hdbscan=0.8.29 \
    datajoint=0.12.9 \
    \
    && /opt/conda/bin/conda install -y -c pytorch -c nvidia pytorch=2.0.0 pytorch-cuda=11.8 \
    \
    && /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

WORKDIR /workspace
RUN chmod -R a+w /workspace

EXPOSE 8888
ADD ./jupyter_lab_config.py /root/.jupyter/
ENTRYPOINT ["jupyter", "lab", "--allow-root"]
