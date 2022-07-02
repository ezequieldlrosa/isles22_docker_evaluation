#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

docker volume create isles22-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="4g" \
        --memory-swap="4g" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/ \
        -v isles22-output-$VOLUME_SUFFIX:/output/ \
        isles22

docker run --rm \
        -v isles22-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/metrics.json | python -m json.tool

docker volume rm isles22-output-$VOLUME_SUFFIX
