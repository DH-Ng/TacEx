ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# # Install whatever you need as additional dependencies.
# Install general development dependencies and remove cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    git-lfs \
    nano \
    libglew-dev \
    freeglut3-dev \
    libeigen3-dev\
    libglib2.0-0 \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# install cuda toolkit for compiling GIPC and the runtime
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

# Install additional dependencies for using TacEx
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/TacEx && \
    pip install -e ."

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
