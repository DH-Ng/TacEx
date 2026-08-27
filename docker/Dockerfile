# For TacEx docker image -> for Isaac ver. 5.1
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

    # libglew-dev \
    # freeglut3-dev \
    # libeigen3-dev \
    # libglib2.0-0 \
    # ncurses-term && \

# set cuda paths
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# requirements compiling cuda code
FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04 AS cuda-devel
# FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04 AS cuda-devel for 3080 gpu


# continue with TacEx container creation
FROM base
COPY --from=cuda-devel /usr/local/cuda /usr/local/cuda

# set extra convenience aliases for IsaacLab
RUN echo "alias run=/isaac-sim/isaac-sim.streaming.sh" >> ${HOME}/.bashrc && \
    echo "alias lab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc

# Remove existing pythons from /usr/bin to avoid confusion
RUN find /usr/bin -name "python*" -exec rm -rf {} +

# Create symlink to IsaacSim python
RUN ln -sf /isaac-sim/python.sh /usr/bin/python && \
    ln -sf /isaac-sim/python.sh /usr/bin/python3

# Set env variables for Isaac 
ENV PATH="/usr/bin:${PATH}"
ENV PYTHONPATH=/isaac-sim/exts/omni.isaac.core:/isaac-sim/exts/omni.isaac.kit

# for Omniverse Streaming Extension choose 1 (deprecated though)
ENV LIVESTREAM=2

# Install specific pip version and and install missing pip packages 
# (Workaround for isaac 5.1, see https://github.com/isaac-sim/IsaacLab/issues/4577#issuecomment-3872804546)
RUN /isaac-sim/python.sh -m pip install pip==23 --force-reinstall && \
    /isaac-sim/python.sh -m pip install setuptools==65 --force-reinstall && \
    /isaac-sim/python.sh -m pip install flatdict==4.0.0 --force-reinstall && \
    /isaac-sim/python.sh -m pip install -e /workspace/isaaclab/source/isaaclab

# For libuipc -> see libuipc artifacts dir
# install cmake 3.26
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

RUN bash -i -c "git config --global --add safe.directory /workspace/tacex"  

# # Set env variables for uipc cmake
# ENV Python_EXECUTABLE=/isaac-sim/kit/python/bin/python3
# ENV Python3_EXECUTABLE=/isaac-sim/kit/python/bin/python3
# ENV Python_ROOT_DIR=/isaac-sim/kit/python
# ENV Python3_ROOT_DIR=/isaac-sim/kit/python
# ENV PATH=/isaac-sim/kit/python/bin:${PATH}

ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11
ENV CMAKE_MAKE_PROGRAM=/usr/bin/make
ENV PATH=/usr/local/cuda/bin:/opt/cmake/bin:/isaac-sim/kit/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# # Install TacEx packages
# # If this does not work, you can comment them out and just call the tacex.sh script after entering the container
# # -> Just don't forget to delete the source/tacex_uipc/build folder for a clean installation of tacex_uipc
# RUN /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex  && \
#     /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex_assets  && \
#     /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex_tasks  && \
#     /isaac-sim/python.sh -m pip install -e ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/tacex_uipc

# Make working directory as the tacex directory
# this is the default directory when the container is run
WORKDIR ${ISAACLAB_EXTENSION_TEMPLATE_PATH}
