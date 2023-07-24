# foundation

Pipeline for neuronal recordings and foundation models.


## Requirements

- Network drive `/mnt/scratch09`
- Docker registry `at-docker.ad.bcm.edu:5000`
- Database account with privileges for `foundation_*` schemas

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
DJ_SUPPORT_FILEPATH_MANAGEMENT=TRUE
DJ_HOST=at-database.ad.bcm.edu
DJ_USER=*
DJ_PASS=*
JUPYTER_TOKEN=*
```

Step 5) Launch the docker container
```
docker compose run -d -p 8880:8888 foundation
```

The jupyter lab environment (protected by the `JUPYTER_TOKEN` variable) should now be accessible at `http://HOSTNAME:8880/`.
