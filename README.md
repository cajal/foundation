# foundation
Foundation Models of the Visual Cortex


## Setup

Step 1) Clone the respository
```
git clone https://github.com/cajal/foundation.git
```

Step 2) Navigate to the `docker` directory
```
cd foundation/docker
```

Step 3) In this directory, create an `.env` file with the following variables at minimum. (Replace * with your own values)
```
DJ_SUPPORT_FILEPATH_MANAGEMENT=TRUE
DJ_HOST=at-database.ad.bcm.edu
DJ_USER=*
DJ_PASS=*
JUPYTER_TOKEN=*
```

Step 4) Launch the docker container
```
docker compose run -d --service-ports foundation
```

The jupyter lab environment should now be accessible at `http://HOSTNAME:8880/`.
