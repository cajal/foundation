#!/bin/bash
set -e

# Usage:
#   ./docker-build.sh        > build only
#   ./docker-build.sh --push > build and push

ENV_PATH="$(dirname "$0")/.env"
PUSH_IMAGE=false

if [[ "$1" == "--push" ]]; then
  PUSH_IMAGE=true
elif [[ -n "$1" ]]; then
  echo "Invalid argument: $1"
  echo "Usage: $0 [--push]"
  exit 1
fi

extract_var() {
  local key=$1
  grep -m 1 -E "^${key}=" "$ENV_PATH" | cut -d '=' -f2-
}

IMAGE_REGISTRY=$(extract_var IMAGE_REGISTRY)
IMAGE_NAMESPACE=$(extract_var IMAGE_NAMESPACE)
IMAGE_NAME=$(extract_var IMAGE_NAME)
IMAGE_TAG=$(extract_var IMAGE_TAG)

IMAGE_REGISTRY=${IMAGE_REGISTRY:-localhost}
IMAGE_NAMESPACE=${IMAGE_NAMESPACE:-cajal}
IMAGE_NAME=${IMAGE_NAME:-foundation}
IMAGE_TAG=${IMAGE_TAG:-dev}

if [[ "$IMAGE_REGISTRY" == "ghcr.io" && "$PUSH_IMAGE" == true ]]; then
  echo "Pushing to ghcr.io is disabled. Use GitHub Actions to publish."
  exit 1
fi

IMAGE_FULL_TAG="${IMAGE_REGISTRY}/${IMAGE_NAMESPACE}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building image ${IMAGE_FULL_TAG}"
docker build -t "${IMAGE_FULL_TAG}" .

if [ "$PUSH_IMAGE" = true ]; then
  echo "Pushing image ${IMAGE_FULL_TAG}"
  docker push "${IMAGE_FULL_TAG}"
fi
