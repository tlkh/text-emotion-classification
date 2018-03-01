#!/usr/bin/env bash

# clear previous run
#rm -R logs && mkdir logs

# run tensorboard on local IP address
tensorboard --logdir=logs --host=35.189.176.14 --port=6009
