# foundation

Pipeline for recordings and foundation models of the visual cortex.


## Setup

Step 1) Make sure `scratch09` is mounted, and the `at-docker.ad.bcm.edu:5000` registry is configured.

Step 2) Clone the respository
```
git clone https://github.com/cajal/foundation.git
```

Step 3) Navigate to the `docker` directory
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
