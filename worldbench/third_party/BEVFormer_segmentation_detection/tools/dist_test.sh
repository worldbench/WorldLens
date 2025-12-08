#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
MODEL_NAME=$3
GPUS=$4
PORT=${PORT:-29503}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT $MODEL_NAME --launcher pytorch ${@:5} --eval bbox
