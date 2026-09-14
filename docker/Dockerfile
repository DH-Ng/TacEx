# TacEx docker image -> Specifically for Isaac ver. 5.1 (and Isaac Lab 2.3.0).
ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base

ARG UBUNTU_VERSION=24.04

# We use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

# Set default RUN shell to bash
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ARG ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV ISAACLAB_EXTENSION_TEMPLATE_PATH=${ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

# Path to the Isaac Lab directory
ARG ISAACLAB_PATH_ARG
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}

USER root

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY ../ ${ISAACLAB_EXTENSION_TEMPLATE_PATH}

# Set up a symbolic link between the Isaac Lab root folder and the TacEx directory
# RUN ln -sf ${ISAACLAB_PATH} ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/_isaac_lab
#-> doesn't do anything, because we mount the complete TacEx directory as volume -> need to add symlink manually after container is build

# # Install whatever you need as additional dependencies.

# Install development dependencies we generally need and remove cache
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gcc-11 g++-11 \
        git \
        wget \
        git-lfs \
        nano \
        tmux \
        pkg-config \
        zip \
        unzip \
        python3-dev \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repository for Ubuntu 24.04 and install Cuda toolkit
# (required to build libuipc)
RUN wget -q \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}

# Make sure that installed cuda toolkit does not mess up Isaac Sim shipped cuda config
ENV LD_LIBRARY_PATH="/isaac-sim/kit/python/lib/python3.11/site-packages/nvidia/cusparselt/lib:/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/cusparselt/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Set extra convenience aliases for IsaacLab
RUN echo "alias run=/isaac-sim/isaac-sim.streaming.sh" >> ${HOME}/.bashrc && \
    echo "alias lab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc

# for Omniverse Streaming Extension choose 1 (deprecated though)
ENV LIVESTREAM=2

# For libuipc -> see libuipc artifacts dir
# Install cmake 3.26
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh && \
    chmod +x cmake-3.26.4-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    ./cmake-3.26.4-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    rm cmake-3.26.4-linux-x86_64.sh
ENV PATH="/opt/cmake/bin:${PATH}"

# Install vcpkg
RUN cd /opt && \
    git clone https://github.com/microsoft/vcpkg.git && \
    cd vcpkg && \
    ./bootstrap-vcpkg.sh -disableMetrics

# Set environment variable for CMake to find vcpkg
ENV VCPKG_ROOT=/opt/vcpkg
ENV CMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake

# Work around libuipc's Python discovery selecting /usr/bin/python3.12.
# Isaac Sim provides Python 3.11 at this path, which is required for
# compatibility with Isaac Sim's Python packages and NumPy extensions.
RUN ln -sf /isaac-sim/kit/python/bin/python3 /usr/bin/python3.12

RUN bash -i -c "git config --global --add safe.directory /workspace/tacex"  

# Better to install TacEx packages after container was build. 
# Otherwise, we run into issues when building libuipc.

# RUN /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex  && \
#     /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex_assets  && \
#     /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex_tasks && \
#     /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex_uipc

# Make working directory as the tacex directory
# this is the default directory when the container is run
WORKDIR ${ISAACLAB_EXTENSION_TEMPLATE_PATH}
