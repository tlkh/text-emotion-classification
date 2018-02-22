#!/usr/bin/env bash

# clear previous run
#rm -R logs && mkdir logs

# run tensorboard on local IP address
tensorboard --logdir=logs --host=10.12.115.65 --port=6006
