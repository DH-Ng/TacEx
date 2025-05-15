ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

ARG ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV ISAACLAB_EXTENSION_TEMPLATE_PATH=${ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

# Path to the Isaac Lab directory
ARG ISAACLAB_PATH_ARG
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Set up a symbolic link between the Isaac Lab root folder and the TacEx directory
# RUN ln -sf ${ISAACLAB_PATH} ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/_isaac_lab
#-> doesnt do anything, because we mount the complete TacEx directory as volume -> need to add symlink manually after container is build

# # Install whatever you need as additional dependencies.

# Install development dependencies we generally need and remove cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    git-lfs \
    nano \
    tmux \
    libglew-dev \
    freeglut3-dev \
    libeigen3-dev \
    libglib2.0-0 \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# # install cuda toolkit for compiling GIPC and the runtime
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get -y install cuda-toolkit-12-4

#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
#    dpkg -i cuda-keyring_1.1-1_all.deb && \
#    apt-get update && apt-get -y install cuda-toolkit-11-8

# set cuda paths 
RUN echo "export CUDA_HOME=/usr/local/cuda" >> ${HOME}/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH" >> ${HOME}/.bashrc && \
    echo "export PATH=$CUDA_HOME/bin:$PATH" >> ${HOME}/.bashrc

# set extra alias for convenience
RUN echo "alias run=/isaac-sim/isaac-sim.streaming.sh" >> ${HOME}/.bashrc && \
    echo "alias lab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "export LIVESTREAM=1" >> ${HOME}/.bashrc

# Install TacEx package
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${ISAACLAB_EXTENSION_TEMPLATE_PATH} && \
    ./tacex.sh -i && \
    git config --global --add safe.directory /workspace/tacex"

# make working directory as the tacex directory
# this is the default directory when the container is run
WORKDIR ${ISAACLAB_EXTENSION_TEMPLATE_PATH}