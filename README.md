# Foundation
Neuronal recordings and Foundation models
This repository covers the training and analysis pipeline used for the foundation model described in [Wang et al., 2025](https://www.nature.com/articles/s41586-025-08829-y). While the code is available for public inspection, it is designed to be used in tandem with lab infrastructure, which is not publicly accessible. Core methods used in this repo are imported from the [FNN](https://github.com/cajal/fnn) repository which can be used to access source code for model architecture, publicly released trained model weights, and tutorials that demonstrate how to fine tune the foundation model to new data.

## Prerequisites
- Docker $\geq$ 20.10
- Docker Compose v2+
- NVIDIA GPU + drivers compatible with CUDA 11.8+
- NVIDIA Container Toolkit
No GPU? You can still run the limited functionality CPU-only service (`foundation-cpu`).

## Installation
### 1. Clone the repository

```
git clone https://github.com/cajal/foundation.git
```

### 2. Navigate to the `docker` directory
```
cd foundation/docker
```

### 3. Create an `.env` file. Add the following lines as needed (replace * with your own values):
For database access:
```
DJ_HOST=*        # database host
DJ_USER=*        # database username
DJ_PASS=*        # database password
```
To configure Jupyter:
```
JUPYTER_TOKEN=*           # your desired password;     if omitted, there is no password prompt
JUPYTER_PORT=*            # your desired port on host; if omitted, defaults to: 8888
```
To customize the image source and tag:
```
IMAGE_REGISTRY=*   # your local registry;            if omitted, defaults to: ghcr.io
IMAGE_NAMESPACE=*  # desired namespace;              if omitted, defaults to: cajal
IMAGE_NAME=*       # desired image name;             if omitted, defaults to: foundation
IMAGE_TAG=*        # desired image tag (e.g. dev);   if omitted, defaults to: latest
```

### 4. Configure the Docker image

**Default image:**
By default, docker compose will use the image from the source that is resolved from the `IMAGE` environment variables in the `.env` file. If the values are not found in the `.env` file, then the default values will resolve to: `ghcr.io/cajal/foundation:latest`. This image represents the latest developments to `docker/Dockerfile`.

**Legacy image:**
To use the image that is closest to that used to train the foundation model released in [Wang et al., 2025](https://www.nature.com/articles/s41586-025-08829-y) (built from `docker/legacy/Dockerfile`), use the image tag `ghcr.io/cajal/foundation:nature_v1`. This can be specified in the `.env` file as follows:
```
IMAGE_REGISTRY=ghcr.io
IMAGE_NAMESPACE=cajal
IMAGE_NAME=foundation
IMAGE_TAG=nature_v1
```
Note: The `foundation:nature_v1` image installed`scipy==1.10.1` with `scikit-image==0.20.0` and `Python 3.8`, a combination that the newer libmamba solver rejects. This package incompatibility is resolved in the `foundation:latest` image.

**Build the image:**
To build the image locally, run `docker/docker-build.sh`. This script will build the image from `docker/Dockerfile` and tag it according to the `IMAGE` environment variables in the `.env` file. If you have push access to the specified registry you can use the `--push` option to build & push (disabled for ghcr.io).

### 5. Launch the Docker container
Navigate to the `docker` directory.
Run:
```
docker compose up -d <service>
```
Where, `<service>` is one of:
`foundation` - for GPU configuration, see requirements above.
`foundation-cpu` - for CPU-only configuration

### 6. Access the container
The jupyter lab environment (protected by the `JUPYTER_TOKEN` `.env` variable) should now be accessible at `http://<your-IP>:8888/`, if you used the default `JUPYTER_PORT` configuration. Otherwise replace `8888` with the value specified at `JUPYTER_PORT` in your `.env` file. Replace `<your-IP>` with the container host's IP address.