# Foundation

Neuronal recordings and Foundation models

This repository covers the training and analysis pipeline used for the foundation model described in [Wang et al., 2025](https://www.nature.com/articles/s41586-025-08829-y).
For accessing source code, trained model weights, or to train your own version of the model, please see: https://github.com/cajal/fnn


## Requirements

- Database account `at-database.ad.bcm.edu:3306`
- Docker registry `at-docker.ad.bcm.edu:5000`
- Network drive `/mnt/scratch09`


## Setup

Step 1) Clone the respository
```
git clone https://github.com/cajal/foundation.git
```

Step 2) Navigate to the `docker` directory
```
cd foundation/docker
```

Step 3) Create an `.env` file with the following variables at minimum. (Replace * with your own values)
```
DJ_HOST=at-database.ad.bcm.edu
DJ_USER=*
DJ_PASS=*
JUPYTER_TOKEN=*
```

Step 4) Launch the docker container
```
docker compose run -d -p 8880:8888 foundation
```

The jupyter lab environment (protected by the `JUPYTER_TOKEN` variable) should now be accessible at `http://HOSTNAME:8880/`.
