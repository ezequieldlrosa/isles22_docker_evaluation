#!/usr/bin/env bash

./build.sh

docker save isles22 | gzip -c > isles22.tar.gz
