# TacEx - Tactile Extension for Isaac Sim/Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

 <!--todo Add description for TacEx  -->
This repository serves as a template for building projects or extensions based on Isaac Lab. It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

The structure is similar to the one from the IsaacLab repo.


**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
>[!CAUTION]
>Make sure that you have [git-lfs](https://git-lfs.com/) installed, otherwise the USD assets won't work!
>>Guide: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage


```bash
git clone https://github.com/Duuuc/TacEx---Tactile-Extension.git
```
- TacEx consists of multiple extensions. To install them all use the `tacex_install.sh` script:
```bash
./tacex.sh -i
```
<!-- 
- Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/TacEx
``` -->

- Verify that the extension is correctly installed by running the following command:

```bash
isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-Base-v1 --num_envs 1024 --enable_cameras
```

## Docker setup (recommended)

### Building Isaac Lab Base Image

Currently, there is no Docker image for Isaac Lab publicly available. Hence, you'd need to build the docker image
for Isaac Lab locally by following the steps for building the container [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#docker-guide).

>[!NOTE]
>**Prerequisites**
> - Have Nvidia Drivers, Docker and the Nvidia Container Toolkit, see [container setup](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html#container-setup)
>   - Helpful for Driver Installation on Linux [Driver Installation Linux](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q1-how-to-install-a-driver)
> - Setup Nvidia API key:
>   - Install the ngc client https://org.ngc.nvidia.com/setup/installers/cli (for the config just press enter twice. i.e. use the default config)
>   - For setting up the API key: [api key setup](https://org.ngc.nvidia.com/setup/api-key)
![image](https://github.com/user-attachments/assets/f773bcc2-fed0-4266-9fb2-10e23b9f874f)

**In a nutshell**:  
```bash
# clone the IsaacLab repo
git clone https://github.com/isaac-sim/IsaacLab.git
cd ./IsaacLab
# activate the script to build the docker container
./docker/container.py start 
```

Once you have built the base Isaac Lab image, you can check it exists by doing:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   18.9GB
```

### Building TacEx Image for Isaac Lab
After building the Isaac Lab container, you can build the docker container for this project. It is called `isaac-lab-tacex`. 

```bash
# assuming you are in the root directory of this Repo
./docker/container.py build # this command also starts the container in the background
```
>[!NOTE]
>For simplicity we use the container script from Isaac Lab (slightly modified) for building, starting, entering and stopping the container. 
>Additional features, such as different container profiles, are currently not supported here.

You can verify the image is built successfully using the same command as earlier:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# isaac-lab-tacex                  latest    1eaf4cd9dce3   20 seconds ago      17.1GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   16.9GB
```
>[!NOTE]
> If you don't want to use the base image `isaac-lab-base` for the TacEx container, then you need to adjust the name `ISAACLAB_BASE_IMAGE` in the `docker/.env.base` file of this repository.

### Running the container

If you just want to start the container, you can do this with:

```bash
./docker/container.py start
```
This will start the services defined in our `docker-compose.yaml` file in detached mode.

To enter the container use

```bash
./docker/container.py enter
```

and to stop it
```bash
./docker/container.py stop
```

> [!TIP]
> The container script can be found in `./docker/container.py`. Just setup an alias in your `~/.bashrc` file for conveniently calling it. 
> For example via `alias tacex="/path_to_repo/docker/container.py"`.

### Interacting with a running container

If you want to run commands inside the running container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} isaac-lab-template /bin/bash
```

### Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the containers, but keeps the images.

### Livestreaming
To see the IsaacSim GUI use the [streaming client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html).

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

<!-- - make sure that env variables `${ISAACLAB_PATH}` and `ISAACLAB_EXTENSION_TEMPLATE_PATH` are set properly.
This is done automatically in the docker setup. You can set it manually like this:
```bash
export ISAACLAB_PATH="/path_to/isaaclab"
export ISAACLAB_EXTENSION_TEMPLATE_PATH="/path_to/tacex"
```  -->

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation. 
> For the docker setup this is `/workspace/isaaclab/_isaac_sim`

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.

If IsaacLab and IsaacSim python modules are not indexed correctly (i.e., the IDE cannot find them when being in the code editor), then
you need to adjust the `"python.analysis.extraPaths"` in the `.vscode/settings.json` file.


>[!NOTE]
>In the docker setup you can use the `/docker/settings.json` file of this repo to replace the one in the `.vscode` folder for indexing
>IsaacLab and TacEx modules. (Idk why the ones from IsaacSim don't work right now.)
><!-- - You first need to create a symbolic link to the IsaacLab files:
```bash
# in a terminal inside the tacex docker container
ln -sf ${ISAACLAB_PATH} ${ISAACLAB_EXTENSION_TEMPLATE_PATH}/_isaac_lab
``` -->
>

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/TacEx"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/TacEx/TacEx/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `IsaacLabExtensionTemplate/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

